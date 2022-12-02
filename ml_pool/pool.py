from typing import Any, Optional
from multiprocessing import Queue, Manager
import threading
import time
from queue import Full
import uuid
import datetime

from ml_pool.logger import get_logger
from ml_pool.custom_types import (
    ScoreModelCallable,
    SharedDict,
    OptionalArgs,
    OptionalKwargs,
    MLModels,
)
from ml_pool.config import Config
from ml_pool.worker import MLWorker
from ml_pool.messages import JobMessage
from ml_pool.exceptions import (
    UserProvidedCallableFailedError,
    JobWithSuchIDDoesntExistError,
    UserProvidedCallableError,
)
from ml_pool.utils import get_new_job_id


# TODO: Can a worker process just hang? Should I kill it manually every now
#       and then?


__all__ = ["MLPool"]


logger = get_logger("ml_pool")


class MLPool:
    """
    TBA
    """

    def __init__(
        self,
        models_to_load: MLModels,
        nb_workers: int = Config.WORKERS_COUNT,
        message_queue_size: int = Config.MESSAGE_QUEUE_SIZE,
    ) -> None:
        self._verify_provided_callables(models_to_load)
        self._models_to_load = models_to_load

        self._nb_workers = max(1, nb_workers)
        self._message_queue: "Queue[JobMessage]" = Queue(
            maxsize=max(Config.DEFAULT_MIN_QUEUE_SIZE, message_queue_size)
        )
        self._manager = Manager()
        self._shared_dict: SharedDict = self._manager.dict()
        self._shared_dict[Config.CANCELLED_JOBS_KEY_NAME] = set()
        self._scheduled_job_ids: set[uuid.UUID] = set()
        self._workers: list[MLWorker] = self._start_workers(nb_workers)
        (
            self._background_threads,
            self._stop_events,
        ) = self._start_background_threads()
        self._workers_healthy = True
        self._workers_exception = None
        self._pool_running = True

        time.sleep(1.0)  # Time to spin up workers, load the models etc
        logger.info(f"MLPool initialised. {nb_workers} workers spun up")

    def create_job(
        self,
        *,
        score_model_function: ScoreModelCallable,
        model_name: str,
        args: OptionalArgs = None,
        kwargs: OptionalKwargs = None,
        wait_if_full: bool = True,
    ) -> Optional[uuid.UUID]:
        """
        Creates a scoring job on the pool.

        score_model_function - a callable which accepts the model (model_name)
        as the first parameter and args, kwargs to run on the pool.

        wait_if_full - the pool has certain capacity, if its full and the flag
        is set, the call is blocking. Set to False to avoid blocking the caller
        """
        # TODO: Check queue size in advance (fails on MacOS lmao)
        # TODO: Check if pickable (could be sent to the process)

        self._ensure_workers_healthy(
            message_if_unhealthy="Cannot create new job, workers failed"
        )
        if not callable(score_model_function):
            raise ValueError(
                "score_model_function must be a callable accepting load model "
                "(model_name) and args, kwargs as parameters."
            )
        if not model_name or model_name not in self._models_to_load:
            raise ValueError(
                f"Incorrect model name provided. "
                f"Available models: {self._models_to_load.keys()}"
            )
        job_id = get_new_job_id()
        job = JobMessage(
            created_at=datetime.datetime.now(),
            message_id=job_id,
            user_func=score_model_function,
            model_name=model_name,
            args=args,
            kwargs=kwargs,
        )
        created = self._enqueue_new_job(job, wait_if_full)
        return job_id if created else None

    def get_result(
        self, *, job_id: uuid.UUID, wait_if_unavailable: bool = True
    ) -> Any:
        """
        Get result of the scoring job created using its id

        If the result for the job_id is not available at the moment and the
        wait_if_unavailable flag is set, the method blocks the caller until
        the result becomes available.
        """
        if job_id not in self._scheduled_job_ids:
            raise JobWithSuchIDDoesntExistError(
                f"Job with id {job_id} doesn't exist or expired"
            )
        if job_id in self._shared_dict:
            return self._retrieve_job_result(job_id)

        if not wait_if_unavailable:
            return

        # If workers failed, we won't be able to retrieve the result
        self._ensure_workers_healthy(
            message_if_unhealthy="Cannot get job results, workers failed"
        )
        while True:
            time.sleep(0.01)  # 10 ms
            if job_id in self._shared_dict:
                return self._retrieve_job_result(job_id)

    def cancel_job(self, job_id: uuid.UUID) -> None:
        """
        Cancel a job

        Might be beneficial when, say, a WS client disconnects, so there is
        nobody to return the result to --> cancel the scoring job.
        """
        if job_id not in self._scheduled_job_ids:
            raise JobWithSuchIDDoesntExistError(
                f"Cannot cancel the job that doesn't exist or expired"
            )
        # The job has already been complete, retrieve the result
        if job_id in self._shared_dict:
            _ = self._retrieve_job_result(job_id)
            return

        # The workers haven't processed the job yet, signal them
        self._cancel_job(job_id)
        logger.debug(f"Cancelled job {job_id}")

    def _start_workers(self, nb_workers: int) -> list[MLWorker]:
        """
        Start MLWorkers running on other cores which will run the user provided
        code
        """
        workers = []
        for _ in range(nb_workers):
            worker = MLWorker(
                shared_dict=self._shared_dict,
                message_queue=self._message_queue,
                ml_models=self._models_to_load,
            )
            worker.start()
            workers.append(worker)
        return workers

    def _enqueue_new_job(self, job: JobMessage, wait_if_full: bool) -> bool:
        """
        Try putting the job in the message queue. The queue might be full, so
        depending on whether the wait_if_full flag is set this call could be
        blocking
        """
        warning_shown = False
        while True:
            try:
                self._message_queue.put_nowait(job)
            except Full:
                if not warning_shown:
                    logger.warning(
                        f"Inner message queue is full. Consider increasing "
                        f"the size or slow down"
                    )
                    warning_shown = True

                if not wait_if_full:
                    return False

                time.sleep(0.01)
            else:
                self._scheduled_job_ids.add(job.message_id)
                break
        logger.info(f"New job created, id: {job.message_id}")
        return True

    def _cancel_job(self, job_id: uuid.UUID) -> None:
        """
        Cancel an active job by adding it to the shared set of jobs to cancel,
        that each MLWorker checks before executing a newly received task.
        """
        self._scheduled_job_ids.remove(job_id)
        self._shared_dict[Config.CANCELLED_JOBS_KEY_NAME].add(job_id)

    def _retrieve_job_result(self, job_id: uuid.UUID) -> Any:
        """
        Retrieve results from the shared dict and clean
        """
        result, _ = self._shared_dict[job_id]
        del self._shared_dict[job_id]
        self._scheduled_job_ids.remove(job_id)
        return result

    def _start_background_threads(
        self,
    ) -> tuple[list[threading.Thread], list[threading.Event]]:
        """
        Starts threads that do some background tasks
        """
        threads, stop_events = [], []

        # Workers monitoring thread
        workers_monitor_event = threading.Event()
        workers_monitor_thread = threading.Thread(
            target=self._monitor_workers,
            args=(workers_monitor_event, Config.MONITOR_THREAD_SLEEP_TIME),
            name="Workers health monitor",
        )
        workers_monitor_thread.start()
        threads.append(workers_monitor_thread)
        stop_events.append(workers_monitor_event)

        # Shared dict cleaning thread
        shared_dict_cleaner_event = threading.Event()
        shared_dict_cleaner_thread = threading.Thread(
            target=self._clean_shared_dict,
            args=(shared_dict_cleaner_event, Config.CLEANER_THREAD_SLEEP_TIME),
            name="Shared dict cleaner",
        )
        shared_dict_cleaner_thread.start()
        threads.append(shared_dict_cleaner_thread)
        stop_events.append(shared_dict_cleaner_event)

        return threads, stop_events

    def _monitor_workers(
        self, stop_event: threading.Event, sleep_time: float = 0.1
    ) -> None:
        """
        Ensures the required number of healthy MLWorkers
        """
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
                    break

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

    def _clean_shared_dict(
        self, stop_event: threading.Event, sleep_time: float = 0.1
    ) -> None:
        """
        Removes results from the shared dict that have been there for too long,
        the caller probably will never retrieve them (WS disconnected etc) but
        they keep consuming memory
        """
        logger.debug("Result dict cleaner thread started")

        while not stop_event.is_set():
            time.sleep(sleep_time)
            for key, value in self._shared_dict.items():
                if not isinstance(key, uuid.UUID):
                    continue

                processed_at, _ = value
                if (
                    datetime.datetime.now() - processed_at
                ).total_seconds() > Config.RESULT_TTL:
                    self._retrieve_job_result(key)
                    logger.debug(f"Job {key} expired, cleaned")

        logger.debug("Result dict cleaner thread stopped")

    def _ensure_workers_healthy(self, message_if_unhealthy: str = "") -> None:
        """
        Checks if the MLWorkers are healthy before the user can do anything
        to the pool
        """
        if not self._workers_healthy:
            if message_if_unhealthy:
                logger.error(message_if_unhealthy)
            self.shutdown()
            raise self._workers_exception  # type: ignore

    @staticmethod
    def _verify_provided_callables(ml_models: MLModels) -> None:
        """
        Checks if all user provided callables are actual functions that can
        be called by the MLWorkers
        """
        for model_name, load_model_callable in ml_models.items():
            if not callable(load_model_callable):
                raise UserProvidedCallableError(
                    f"Callable to load model {model_name} is not a callable"
                )

    def __enter__(self) -> "MLPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self) -> None:
        # TODO: Joins need timeout

        if not self._pool_running:
            return

        # Stop background threads
        for event in self._stop_events:
            event.set()
        for thread in self._background_threads:
            thread.join()
            logger.debug(f"Thread {thread.name} stopped")

        # Stop MLWorkers
        for worker in self._workers:
            worker.terminate()
        for worker in self._workers:
            worker.join()
        logger.debug("MLWorkers stopped")

        self._manager.shutdown()
        self._manager.join()
        logger.debug("Manager process stopped")

        self._pool_running = False
        logger.info("MLPool shutdown gracefully")
