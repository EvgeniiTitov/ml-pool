from typing import Any, Optional
from multiprocessing import Queue, Manager
import threading
import time

from ml_pool.logger import get_logger
from ml_pool.custom_types import LoadModelCallable, ScoreModelCallable
from ml_pool.config import Config
from ml_pool.worker import MLWorker
from ml_pool.messages import JobMessage
from ml_pool.exceptions import UserProvidedCallableFailedError


# TODO: How to communicate exception (user fucked up) to the main thread?


__all__ = ["MLPool"]


logger = get_logger("ml_pool")


class MLPool:

    def __init__(
        self,
        load_model_func: LoadModelCallable,
        score_model_func: ScoreModelCallable,
        nb_workers: int = Config.WORKERS_COUNT,
        message_queue_size: int = Config.MESSAGE_QUEUE_SIZE
    ) -> None:
        self._nb_workers = nb_workers
        self._load_model_func = load_model_func
        self._score_model_func = score_model_func

        self._message_queue: "Queue[JobMessage]" = Queue(message_queue_size)
        self._manager = Manager()
        self._result_dict = self._manager.dict()

        self._workers: list[MLWorker] = self._start_workers(nb_workers)

        self._monitor_thread_stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_workers,
            args=(
                self._monitor_thread_stop_event,
                Config.MONITOR_THREAD_SLEEP_TIME
            )
        )
        self._monitor_thread.start()
        logger.info("MLPool initialised")

    def _start_workers(self, nb_workers: int) -> list[MLWorker]:
        workers = []
        for _ in range(nb_workers):
            worker = MLWorker(
                result_dict=self._result_dict,
                message_queue=self._message_queue,
                load_model_func=self._load_model_func,
                score_model_func=self._score_model_func
            )
            worker.start()
            workers.append(worker)
        return workers

    def score_model(self, *args, **kwargs) -> Any:
        # TODO: Model scoring
        # TODO: Proper example using a model
        pass

    def shutdown(self, exception: Optional[Exception] = None) -> None:
        self._monitor_thread_stop_event.set()
        self._monitor_thread.join()
        logger.debug("Workers monitoring thread joined")

        for worker in self._workers:
            worker.terminate()
        for worker in self._workers:
            worker.join()
        logger.debug("Worker joined")

        if exception:
            raise exception

    def _monitor_workers(
        self, stop_event: threading.Event, sleep_time: int = 1.0
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
                    not worker.is_alive() and
                    worker.exitcode == Config.USER_CODE_FAILED_EXIT_CODE
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
