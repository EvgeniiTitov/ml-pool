from multiprocessing import Process, Queue
import sys
import os

from ml_pool.logger import get_logger
from ml_pool.custom_types import SharedDict, MLModels, LoadedMLModels
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
        shared_dict: SharedDict,
        message_queue: Queue,
        ml_models: MLModels,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._shared_dict: SharedDict = shared_dict
        self._message_queue: Queue["JobMessage"] = message_queue
        self._ml_models = ml_models

    def run(self) -> None:
        pid = os.getpid()
        logger.info(f"MLWorker {pid} started")

        loaded_ml_models: LoadedMLModels = self._load_models(self._ml_models)
        logger.info(f"MLWorker {pid} loaded models: {self._ml_models.keys()}")

        while True:
            message: JobMessage = self._message_queue.get()
            job_id = message.message_id
            func = message.user_func
            model_name = message.model_name
            args = message.args or []
            kwargs = message.kwargs or {}

            # Check if the job's been cancelled by the caller
            if job_id in self._shared_dict[Config.CANCELLED_JOBS_KEY_NAME]:
                self._shared_dict[Config.CANCELLED_JOBS_KEY_NAME].remove(
                    job_id
                )
                continue

            if model_name not in loaded_ml_models:
                logger.error(
                    f"Model {model_name} wasn't loaded by the MLWorker, "
                    f"cannot pass it to callable {func.__name__}"
                )
                sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

            try:
                result = func(
                    loaded_ml_models[model_name], *args, **kwargs
                )  # type: ignore
            except Exception as e:
                logger.error(
                    f"User provided callable {func.__name__} called with "
                    f"model {model_name}, args {args}, kwargs {kwargs} failed"
                    f"with error: {e}"
                )
                sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

            logger.debug(
                f"MLWorker {pid} ran callable {func.__name__} with model "
                f"{model_name}, args {args}, kwargs {kwargs}. Result: {result}"
            )
            self._shared_dict[job_id] = result

    @staticmethod
    def _load_models(ml_models: MLModels) -> LoadedMLModels:
        loaded_models = {}

        for model_name, load_model_callable in ml_models.items():
            try:
                loaded_model = load_model_callable()  # type: ignore
            except Exception as e:
                logger.error(
                    f"Failed while loading model {model_name} using "
                    f"{load_model_callable.__name__} callable. Error: {e}"
                )
                sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

            if not loaded_model:
                logger.error(
                    f"Provided callable {load_model_callable.__name__} to "
                    f"load model {model_name} didn't return a valid object"
                )
                sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)
            loaded_models[model_name] = loaded_model

        return loaded_models
