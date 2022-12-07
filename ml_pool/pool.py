import asyncio
from typing import Any, Optional, List, Tuple
import threading
import time
from queue import Full
import uuid
import datetime

from ml_pool.logger import get_logger
from ml_pool.custom_types import (
    ScoreModelCallable,
    ResultDict,
    OptionalArgs,
    OptionalKwargs,
    MLModels,
    CancelledDict,
)
from ml_pool.config import Config
from ml_pool.worker import MLWorker
from ml_pool.messages import JobMessage
from ml_pool.exceptions import (
    MLWorkerFailedBecauseOfUserProvidedCodeError,
    JobWithSuchIDDoesntExistError,
    UserProvidedCallableError,
)
from ml_pool.utils import get_new_job_id, context, get_manager


__all__ = ["MLPool"]


logger = get_logger("ml_pool")


class MLPool:
    def __init__(
        self,
        models_to_load: MLModels,
        nb_workers: int = Config.WORKERS_COUNT,
        message_queue_size: int = Config.MESSAGE_QUEUE_SIZE,
        result_ttl: float = Config.RESULT_TTL,
    ) -> None:
        self._verify_provided_callables(models_to_load)
        self._models_to_load = models_to_load

        # Create message queue to send jobs to MLWorkers
        self._nb_workers = max(1, nb_workers)
        self._message_queue = context.Queue(  # type: ignore
            maxsize=max(Config.DEFAULT_MIN_QUEUE_SIZE, message_queue_size)
        )
        # Create a manager and shared dictionaries for bidirectional data
        # exchange between the pool and ML Workers
        self._manager = get_manager()
        self._result_dict: ResultDict = self._manager.dict()
        self._cancel_dict: CancelledDict = self._manager.dict()
        self._scheduled_job_ids: set[uuid.UUID] = set()

        self._workers: List[MLWorker] = self._start_workers(nb_workers)
        (
            self._background_threads,
            self._stop_events,
        ) = self._start_background_threads()

        self._result_ttl = max(result_ttl, Config.RESULT_TTL)
        self._workers_healthy = True
        self._worker_error_description = ""
        self._workers_exit_code = None

        self._pool_running = True
        time.sleep(1.0)  # Time to spin up workers, load the models etc
        logger.info(f"MLPool initialised. {nb_workers} workers spun up")

    # ---------------------------- Public methods ----------------------------
    def create_job(
        self,
        score_model_function: ScoreModelCallable,
        model_name: str,
        args: OptionalArgs = None,
        kwargs: OptionalKwargs = None,
        *,
        wait_if_full: bool = True,
    ) -> Optional[uuid.UUID]:
        """
        Creates a scoring job on the pool.

        score_model_function - a callable which accepts the model (model_name)
        as the first parameter and args, kwargs to run on the pool.

        wait_if_full - the pool has certain capacity, if its full and the flag
        is set, the call is blocking. Set to False to avoid blocking the caller
        """
        self._validate_args_for_create_job(score_model_function, model_name)
        job_id = get_new_job_id()
        job = JobMessage(
            message_id=job_id,
            user_func=score_model_function,
            model_name=model_name,
            args=args,
            kwargs=kwargs,
        )
        enqueued = self._enqueue_new_job(job, wait_if_full)
        return job_id if enqueued else None

    async def create_job_async(
        self,
        score_model_function: ScoreModelCallable,
        model_name: str,
        args: OptionalArgs = None,
        kwargs: OptionalKwargs = None,
    ):
        """
        Similar to create_job() function, but it doesn't block the event loop
        if the queue is pool is full.
        """
        self._validate_args_for_create_job(score_model_function, model_name)
        job_id = get_new_job_id()
        job = JobMessage(
            message_id=job_id,
            user_func=score_model_function,
            model_name=model_name,
            args=args,
            kwargs=kwargs,
        )
        while True:
            enqueued = self._enqueue_new_job(job, wait_if_full=False)
            if not enqueued:
                await asyncio.sleep(0.005)
            else:
                return job_id

    def get_result(
        self, job_id: uuid.UUID, *, wait_if_unavailable: bool = True
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
        if job_id in self._result_dict:
            return self._retrieve_job_result(job_id)

        if not wait_if_unavailable:
            return

        # If workers failed, we won't be able to retrieve the result
        self._ensure_workers_healthy(
            message_if_unhealthy="Cannot get job results, workers failed"
        )
        while True:
            time.sleep(0.01)  # 10 ms
            if job_id in self._result_dict:
                return self._retrieve_job_result(job_id)

    async def get_result_async(self, job_id: uuid.UUID) -> Any:
        """
        Similar to the get_result() function, but it doesn't block the event
        loop if the result is not available yet
        """
        if job_id not in self._scheduled_job_ids:
            raise JobWithSuchIDDoesntExistError(
                f"Job with id {job_id} doesn't exist or expired"
            )
        if job_id in self._result_dict:
            return self._retrieve_job_result(job_id)

        # If workers failed, we won't be able to retrieve the result
        self._ensure_workers_healthy(
            message_if_unhealthy="Cannot get job results, workers failed"
        )
        while True:
            await asyncio.sleep(0.005)
            if job_id in self._result_dict:
                return self._retrieve_job_result(job_id)

    def cancel_job(self, job_id: uuid.UUID) -> None:
        """
        Cancel a job

        Might be beneficial when, say, a WS client disconnects, so there is
        nobody to return the result to --> cancel the scoring job.
        """
        if job_id not in self._scheduled_job_ids:
            logger.info(
                f"Job with id {job_id} was never scheduled or completed"
            )
            return

        # The job has already been complete, retrieve the result
        if job_id in self._result_dict:
            _ = self._retrieve_job_result(job_id)
            return

        # The workers haven't processed the job yet, signal them
        self._cancel_job(job_id)
        logger.debug(f"Cancelled job {job_id}")

    # ------------------------------ Private methods -------------------------
    def _start_workers(self, nb_workers: int) -> List[MLWorker]:
        """
        Start MLWorkers running on other cores which will run the user provided
        code
        """
        workers = []
        for _ in range(nb_workers):
            worker = MLWorker(
                message_queue=self._message_queue,
                result_dict=self._result_dict,
                cancelled_dict=self._cancel_dict,
                ml_models=self._models_to_load,
                daemon=True,
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
        self._cancel_dict[job_id] = None  # Using as a set (hacky)

    def _retrieve_job_result(self, job_id: uuid.UUID) -> Any:
        """
        Retrieve results from the shared dict and clean
        """
        _, result = self._result_dict[job_id]
        del self._result_dict[job_id]
        self._scheduled_job_ids.remove(job_id)
        return result

    def _validate_args_for_create_job(
        self, score_model_function: ScoreModelCallable, model_name: str
    ) -> None:
        self._ensure_workers_healthy(
            message_if_unhealthy="Cannot create new job, workers failed"
        )
        if not callable(score_model_function):
            raise ValueError(
                "score_model_function must be a callable accepting load model "
                "(model_name) and args, kwargs as parameters"
            )
        if asyncio.iscoroutinefunction(score_model_function):
            raise ValueError(
                "score_must_function must be a function, not a coroutine"
            )
        if not model_name or model_name not in self._models_to_load:
            raise ValueError(
                f"Incorrect model name provided. "
                f"Available models: {list(self._models_to_load.keys())}"
            )

    def _start_background_threads(
        self,
    ) -> Tuple[List[threading.Thread], List[threading.Event]]:
        """
        Starts threads that do some background tasks
        """
        threads, stop_events = [], []

        # Workers monitoring thread
        workers_monitor_event = threading.Event()
        workers_monitor_thread = threading.Thread(
            target=self._monitor_workers,
            args=(workers_monitor_event, Config.MONITOR_THREAD_FREQUENCY),
            name="Workers health monitor",
            daemon=True,
        )
        workers_monitor_thread.start()
        threads.append(workers_monitor_thread)
        stop_events.append(workers_monitor_event)

        # Shared dict cleaning thread
        shared_dict_cleaner_event = threading.Event()
        shared_dict_cleaner_thread = threading.Thread(
            target=self._clean_result_dict,
            args=(shared_dict_cleaner_event, Config.CLEANER_THREAD_FREQUENCY),
            name="Result dict cleaner",
            daemon=True,
        )
        shared_dict_cleaner_thread.start()
        threads.append(shared_dict_cleaner_thread)
        stop_events.append(shared_dict_cleaner_event)

        return threads, stop_events

    def _monitor_workers(
        self,
        stop_event: threading.Event,
        sleep_time: float = Config.MONITOR_THREAD_FREQUENCY,
    ) -> None:
        """
        Ensures the required number of healthy MLWorkers
        """
        logger.debug("Workers monitoring thread started")

        while not stop_event.is_set():
            time.sleep(sleep_time)

            # Check workers in the pool, if a worker failed because of a user
            # provided callable, stop the pool
            healthy_workers = []
            for worker in self._workers:
                if worker.is_alive():
                    healthy_workers.append(worker)
                elif (
                    not worker.is_alive()
                    and worker.exitcode not in Config.CUSTOM_EXIT_CODES_MAPPING
                ):
                    logger.error("MLWorker failed unexpectedly, restarting...")
                elif (
                    not worker.is_alive()
                    and worker.exitcode in Config.CUSTOM_EXIT_CODES_MAPPING
                ):
                    self._workers_healthy = False
                    self._worker_error_description = (
                        Config.CUSTOM_EXIT_CODES_MAPPING[worker.exitcode]
                    )
                    self._workers_exit_code = worker.exitcode  # type: ignore
                    return

            total_healthy = len(healthy_workers)
            if total_healthy < self._nb_workers:
                logger.debug("Fewer workers than required, adding more")
                healthy_workers.extend(
                    self._start_workers(self._nb_workers - total_healthy)
                )
            self._workers = healthy_workers

        logger.debug("Workers monitoring thread stopped")

    def _clean_result_dict(
        self,
        stop_event: threading.Event,
        sleep_time: float = Config.CLEANER_THREAD_FREQUENCY,
    ) -> None:
        """
        Removes results from the result dict that have been there for too long,
        the caller probably will never retrieve them (WS disconnected etc) but
        they keep consuming memory
        """
        logger.debug("Result dict cleaner thread started")
        time.sleep(2.0)

        while not stop_event.is_set():
            time.sleep(sleep_time)

            results_pending = 0
            for job_id, value in self._result_dict.items():
                processed_at, _ = value
                if (
                    datetime.datetime.now() - processed_at
                ).total_seconds() > self._result_ttl:
                    self._retrieve_job_result(job_id)
                    logger.debug(f"Job {job_id} expired, cleaned")
                results_pending += 1

            logger.debug(f"Cleaning ran, pending results {results_pending}")
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
            raise MLWorkerFailedBecauseOfUserProvidedCodeError(
                f"Error description {self._worker_error_description}"
            )

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

    def shutdown(self, timeout: float = Config.SHUTDOWN_JOIN_TIMEOUT) -> None:
        if not self._pool_running:
            return
        logger.info("Shutting down the pool...")

        # Stop background threads
        for event in self._stop_events:
            event.set()
        for thread in self._background_threads:
            thread.join(timeout=timeout)

        # Stop MLWorkers
        for worker in self._workers:
            worker.terminate()
        for worker in self._workers:
            worker.join(timeout=timeout)
        logger.debug("MLWorkers stopped")

        self._pool_running = False
        logger.info("MLPool shutdown gracefully")

    def __enter__(self) -> "MLPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
