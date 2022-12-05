from multiprocessing import Process, Queue, Event
from queue import Empty
import sys
import os
import datetime

from ml_pool.logger import get_logger
from ml_pool.custom_types import (
    ResultDict,
    MLModels,
    LoadedMLModels,
    CancelledDict,
)
from ml_pool.messages import JobMessage
from ml_pool.config import Config


__all__ = ["MLWorker"]


logger = get_logger("ml_worker")


class MLWorker(Process):
    """
    TBA
    """

    GET_MESSAGE_TIMEOUT = 0.1

    def __init__(
        self,
        message_queue: Queue,
        result_dict: ResultDict,
        cancelled_dict: CancelledDict,
        ml_models: MLModels,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._message_queue: Queue["JobMessage"] = message_queue
        self._result_dict: ResultDict = result_dict
        self._cancelled_dict: CancelledDict = cancelled_dict
        self._ml_models = ml_models
        self._stop_event = Event()

    def run(self) -> None:
        pid = os.getpid()
        logger.info(f"MLWorker {pid} started")
        loaded_ml_models: LoadedMLModels = self._load_models(self._ml_models)
        logger.info(
            f"MLWorker {pid} loaded models: {list(self._ml_models.keys())}"
        )
        while not self._stop_event.is_set():
            try:
                message: JobMessage = self._message_queue.get(
                    timeout=MLWorker.GET_MESSAGE_TIMEOUT
                )
            except Empty:
                continue

            job_id = message.message_id
            func = message.user_func
            model_name = message.model_name
            args = message.args or []
            kwargs = message.kwargs or {}

            # Check if the job's been cancelled by the caller
            if job_id in self._cancelled_dict:
                del self._cancelled_dict[job_id]
                logger.debug(
                    f"MLWorker {pid} received job that was cancelled. Skipped"
                )
                continue

            try:
                result = func(
                    loaded_ml_models[model_name], *args, **kwargs
                )  # type: ignore
            except Exception as e:
                logger.error(
                    f"User provided callable {func.__name__} called with "
                    f"model {model_name}, args {args}, kwargs {kwargs} failed "
                    f"with error: {e}",
                )
                sys.exit(Config.SCORE_MODEL_CALLABLE_FAILED)

            logger.debug(
                f"MLWorker {pid} ran callable {func.__name__} with model "
                f"{model_name}, args {args}, kwargs {kwargs}. "
                f"Result: {result}"  # noqa
            )
            self._result_dict[job_id] = (datetime.datetime.now(), result)

        logger.debug(f"MLWorker {pid} was stopped gracefully")

    def initiate_stop(self):
        self._stop_event.set()

    @staticmethod
    def _load_models(ml_models: MLModels) -> LoadedMLModels:
        loaded_models = {}
        for model_name, load_model_callable in ml_models.items():
            try:
                loaded_model = load_model_callable()  # type: ignore
            except Exception as e:
                logger.error(
                    f"Failed while loading model {model_name} using "
                    f"{load_model_callable} callable. Error: {e}"
                )
                sys.exit(Config.LOAD_MODEL_CALLABLE_FAILED)

            if not loaded_model:  # noqa
                logger.error(
                    f"Provided callable {load_model_callable} to "
                    f"load model {model_name} didn't return a valid object. "
                    f"Expected value is a single model instance",
                )
                sys.exit(Config.LOAD_MODEL_CALLABLE_RETURNED_NOTHING)

            loaded_models[model_name] = loaded_model
        return loaded_models
