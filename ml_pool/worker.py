from multiprocessing import Process, Queue
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
        self.error_message = None

    def run(self) -> None:
        pid = os.getpid()
        logger.info(f"MLWorker {pid} started")
        loaded_ml_models: LoadedMLModels = self._load_models(self._ml_models)
        logger.info(
            f"MLWorker {pid} loaded models: {list(self._ml_models.keys())}"
        )
        while True:
            message: JobMessage = self._message_queue.get()
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

            if model_name not in loaded_ml_models:
                self._fail_gracefully(
                    f"Model {model_name} wasn't loaded by the MLWorker, "
                    f"cannot pass it to callable {func.__name__}",
                    exit_code=Config.UNKNOWN_ML_MODEL_REQUESTED,
                )

            try:
                result = func(
                    loaded_ml_models[model_name], *args, **kwargs
                )  # type: ignore
            except Exception as e:
                self._fail_gracefully(
                    f"User provided callable {func.__name__} called with "
                    f"model {model_name}, args {args}, kwargs {kwargs} failed"
                    f"with error: {e}",
                    exit_code=Config.SCORE_MODEL_CALLABLE_FAILED,
                )

            logger.debug(
                f"MLWorker {pid} ran callable {func.__name__} with model "
                f"{model_name}, args {args}, kwargs {kwargs}. "
                f"Result: {result}"  # noqa
            )
            self._result_dict[job_id] = (datetime.datetime.now(), result)

    def _load_models(self, ml_models: MLModels) -> LoadedMLModels:
        loaded_models = {}

        for model_name, load_model_callable in ml_models.items():
            try:
                loaded_model = load_model_callable()  # type: ignore
            except Exception as e:
                self._fail_gracefully(
                    f"Failed while loading model {model_name} using "
                    f"{load_model_callable.__name__} callable. Error: {e}",
                    exit_code=Config.LOAD_MODEL_CALLABLE_FAILED,
                )
            if not loaded_model:  # noqa
                self._fail_gracefully(
                    f"Provided callable {load_model_callable.__name__} to "
                    f"load model {model_name} didn't return a valid object",
                    exit_code=Config.LOAD_MODEL_CALLABLE_RETURNED_NOTHING,
                )
            loaded_models[model_name] = loaded_model

        return loaded_models

    def _fail_gracefully(self, error_message: str, exit_code: int) -> None:
        logger.error(error_message)
        self.error_message = error_message  # type: ignore
        sys.exit(exit_code)
