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


# TODO: Can a worker process just hang? Should I kill it manually every now
#       and then?


__all__ = ["MLPool"]


logger = get_logger("ml_pool")


class MLPool:
    """
    Manages a pool of MLWorkers running in dedicated processes that execute
    user provided code:
        - load_model_func - a callable that loads an ML model in the worker
          process.
        - score_model_func - a callable that takes as an input the loaded model
          plus any other parameters (*args, **kwargs) and returns the scoring
          result

    The pool does not depend on the implementation of the user provided code,
    it can accept anything as long as the callables signatures are adequate
    """

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

        self._message_queue: "Queue[JobMessage]" = Queue(
            maxsize=max(50, message_queue_size)
        )
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
        self._workers_healthy = True
        self._workers_exception = None
        self._pool_running = True
        time.sleep(1.0)  # Time to spin up workers, load the models etc
        logger.info(f"MLPool initialised. {nb_workers} workers spun up")

    def schedule_model_scoring(self, *args, **kwargs) -> uuid.UUID:
        """
        Use this method to create a model scoring job that will run on a worker
        using the model loaded by calling the load_model_func and passing the
        loaded model + the args and kwargs passed to this method to the
        score_model_func callable.

        The method returns whatever your score_model_func returns.
        """
        # TODO: This blocks the caller until the message gets enqueued. Bad?

        if not self._workers_healthy:
            logger.error("Cannot schedule scoring. Workers raised exception")
            self.shutdown()
            raise self._workers_exception  # type: ignore

        job_id = get_new_job_id()
        job_message = JobMessage(message_id=job_id, args=args, kwargs=kwargs)
        warning_shown = False
        while True:
            try:
                self._message_queue.put_nowait(job_message)
            except Full:
                if not warning_shown:
                    logger.warning(
                        "Message (job) queue is full, cannot enqueue new "
                        "scoring job. Increase the queue size or slow down"
                    )
                    warning_shown = True
                time.sleep(0.01)  # 10 ms
            else:
                break
        logger.debug(f"New job scheduled, its id {job_id}")
        return job_id

    def get_scoring_result(
        self, job_id: uuid.UUID, wait_if_not_available: bool
    ) -> Any:
        if job_id in self._result_dict:
            return self._retrieve_job_result(job_id)

        if not wait_if_not_available:
            return

        # If workers failed, we won't be able to retrieve the result
        if not self._workers_healthy:
            logger.error(
                "Cannot get scoring results. Workers raised exception"
            )
            self.shutdown()
            raise self._workers_exception  # type: ignore

        while True:
            time.sleep(0.01)
            if job_id in self._result_dict:
                return self._retrieve_job_result(job_id)

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

    def _retrieve_job_result(self, job_id: uuid.UUID) -> Any:
        result = self._result_dict[job_id]
        del self._result_dict[job_id]
        return result

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
                    self._workers_healthy = False
                    exception = UserProvidedCallableFailedError(
                        "User provided callable threw exception in worker"
                    )
                    self._workers_exception = exception  # type: ignore
                    raise exception

            total_healthy = len(healthy_workers)
            if total_healthy < self._nb_workers:
                logger.info(
                    "Fewer workers than required, adding new to the pool"
                )
                healthy_workers.extend(
                    self._start_workers(self._nb_workers - total_healthy)
                )

            self._workers = healthy_workers
        logger.debug("Workers monitoring thread stopped")

    def shutdown(self) -> None:
        if not self._pool_running:
            return

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

        self._pool_running = False
        logger.info("MLPool shutdown gracefully")
