from multiprocessing import Process, Queue
import sys
import os

from ml_pool.logger import get_logger
from ml_pool.custom_types import (
    LoadModelCallable,
    ScoreModelCallable,
    MLModel,
    ResultDict,
)
from ml_pool.messages import JobMessage
from ml_pool.config import Config


__all__ = ["MLWorker"]


logger = get_logger("ml_worker")


class MLWorker(Process):
    """
    MLWorker runs in a dedicated process, it loads a ML model using the
    passed load_model_func callable.

    MLWorker gets messages (tasks) from the message queue. For each task the
    worker scores the model using the passed score_model_func callable and
    the args and kwargs coming from the message it gets from the queue.

    Each message has a unique id. Upon successful model scoring, the result
    is saved to the shared dictionary (multiprocessing.Manager.dict) using
    the unique id as a key and the result as a value.

    If the user provided code fails (either model loading or scoring), the
    process returns a specific status code, signalling the pool to raise
    an exception
    """

    def __init__(
        self,
        result_dict: ResultDict,
        message_queue: Queue,
        load_model_func: LoadModelCallable,
        score_model_func: ScoreModelCallable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._result_dict: ResultDict = result_dict
        self._message_queue: Queue["JobMessage"] = message_queue
        self._load_model_func = load_model_func
        self._score_model_func = score_model_func

    def run(self) -> None:
        logger.info(f"MLWorker started, pid {os.getpid()}")

        # Load the model object using the user provided callable
        try:
            model: MLModel = self._load_model_func()  # type: ignore
        except Exception as e:
            logger.error(
                f"User provided load model callable failed. Error: {e}"
            )
            sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

        if not model:
            logger.error(
                "User provided load model callable hasn't returned anything"
            )
            sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

        logger.info(f"MLWorker loaded the model: {model.__class__.__name__}")

        # Start processing messages using the loaded model and the scoring
        # function provided by the user
        while True:
            message: JobMessage = self._message_queue.get()
            job_id = message.message_id
            args = message.args or []
            kwargs = message.kwargs or {}
            try:
                result = self._score_model_func(
                    model, *args, **kwargs
                )  # type: ignore
            except Exception as e:
                logger.error(
                    f"User provided score model callable failed for "
                    f"model {model.__class__.__name__} with args {args}, "
                    f"kwargs {kwargs}. Error: {e}"
                )
                sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

            logger.debug(
                f"MLWorker successfully scored model for id: {job_id}, "
                f"result: {result}"
            )
            self._result_dict[job_id] = result
