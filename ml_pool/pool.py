from typing import Any
from multiprocessing import Queue, Manager
import threading
import time
from queue import Full
import uuid

from ml_pool.logger import get_logger
from ml_pool.custom_types import (
    LoadModelCallable,
    ScoreModelCallable,
    ResultDict,
)
from ml_pool.config import Config
from ml_pool.worker import MLWorker
from ml_pool.messages import JobMessage
from ml_pool.exceptions import UserProvidedCallableFailedError
from ml_pool.utils import get_new_job_id


# TODO: How to communicate exception (user fucked up) to the main thread?
#       Add a flag, that the thread sets alongside the exception. Then before
#       enqueueing a new task/getting results, check if the flag is set (pool
#       closed)


__all__ = ["MLPool"]


logger = get_logger("ml_pool")


class MLPool:
    def __init__(
        self,
        load_model_func: LoadModelCallable,
        score_model_func: ScoreModelCallable,
        nb_workers: int = Config.WORKERS_COUNT,
        message_queue_size: int = Config.MESSAGE_QUEUE_SIZE,
    ) -> None:
        self._nb_workers = max(1, nb_workers)

        if not callable(load_model_func):
            raise UserProvidedCallableFailedError(
                "load_model_func must be callable returning a ML model"
            )
        self._load_model_func = load_model_func

        if not callable(score_model_func):
            raise UserProvidedCallableFailedError(
                "score_model_func must be a callable with signature:"
                "(model, *args, **kwargs) -> Any"
            )
        self._score_model_func = score_model_func

        self._message_queue: "Queue[JobMessage]" = Queue(message_queue_size)
        self._manager = Manager()
        self._result_dict: ResultDict = self._manager.dict()
        self._workers: list[MLWorker] = self._start_workers(nb_workers)
        self._monitor_thread_stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_workers,
            args=(
                self._monitor_thread_stop_event,
                Config.MONITOR_THREAD_SLEEP_TIME,
            ),
        )
        self._monitor_thread.start()
        logger.info(f"MLPool initialised. {nb_workers} span up")

    def _start_workers(self, nb_workers: int) -> list[MLWorker]:
        workers = []
        for _ in range(nb_workers):
            worker = MLWorker(
                result_dict=self._result_dict,
                message_queue=self._message_queue,
                load_model_func=self._load_model_func,
                score_model_func=self._score_model_func,
            )
            worker.start()
            workers.append(worker)
        return workers

    def schedule_model_scoring(self, *args, **kwargs) -> uuid.UUID:
        # TODO: Break the loop at some point? This blocks the process!

        job_id = get_new_job_id()
        job_message = JobMessage(message_id=job_id, args=args, kwargs=kwargs)
        warning_shown = False
        while True:
            try:
                self._message_queue.put_nowait(job_message)
            except Full:
                if not warning_shown:
                    logger.warning("Message (job) queue is full")
                    warning_shown = True
                time.sleep(0.1)
            else:
                break
        logger.debug(f"New job scheduled, its id {job_id}")
        return job_id

    def get_scoring_result(self, job_id: uuid.UUID) -> Any:
        # TODO: Time out option?

        while True:
            if job_id in self._result_dict:
                result = self._result_dict[job_id]
                del self._result_dict[job_id]
                return result
            else:
                time.sleep(0.05)

    def shutdown(self) -> None:
        self._monitor_thread_stop_event.set()
        self._monitor_thread.join()
        logger.debug("Workers monitoring thread stopped")

        for worker in self._workers:
            worker.terminate()
        for worker in self._workers:
            worker.join()
        logger.debug("Workers stopped")

        self._manager.shutdown()
        self._manager.join()
        logger.debug("Manager process stopped")
        logger.info("MLPool shutdown gracefully")

    def _monitor_workers(
        self, stop_event: threading.Event, sleep_time: float = 1.0
    ) -> None:
        """Ensures the required number of healthy processes"""
        logger.debug("Workers monitoring thread started")
        while not stop_event.is_set():
            time.sleep(sleep_time)

            healthy_workers = []
            for worker in self._workers:
                if worker.is_alive():
                    healthy_workers.append(worker)
                elif (
                    not worker.is_alive()
                    and worker.exitcode == Config.USER_CODE_FAILED_EXIT_CODE
                ):
                    raise UserProvidedCallableFailedError(
                        "User provided callable threw exception in worker"
                    )

            total_healthy = len(healthy_workers)
            if total_healthy < self._nb_workers:
                logger.debug("Fewer workers than required, adding")
                healthy_workers.extend(
                    self._start_workers(self._nb_workers - total_healthy)
                )

            self._workers = healthy_workers
        logger.debug("Workers monitoring thread stopped")
