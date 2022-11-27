from multiprocessing import Process, Queue
from typing import Dict
import sys

from ml_pool.logger import get_logger
from ml_pool.custom_types import (
    LoadModelCallable,
    ScoreModelCallable,
    BaseMLModel
)
from ml_pool.messages import JobMessage
from ml_pool.config import Config


__all__ = ["MLWorker"]


logger = get_logger("ml_worker")


class MLWorker(Process):
    """
    MLWorker runs in a dedicated process, it loads an ML model using the
    passed load_model_func callable.

    MLWorker gets messages (tasks) from the message queue. For each task the
    worker scores the model using the passed score_model_func callable and
    the args and kwargs coming from the message it gets from the queue.

    Each message has a unique id. Upon successful model scoring, the result
    is saved to the shared dictionary (multiprocessing.Manager.dict) using
    the unique id as a key and result as a value.

    If the user provided code fails, the process return a specific status code,
    signalling the pool to raise an exception
    """
    def __init__(
        self,
        result_dict: Dict,
        message_queue: Queue,
        load_model_func: LoadModelCallable,
        score_model_func: ScoreModelCallable,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._result_dict = result_dict
        self._message_queue: Queue["JobMessage"] = message_queue
        self._load_model_func = load_model_func
        self._score_model_func = score_model_func
        self.running = False
        logger.info(f"MLWorker initialised")

    def run(self) -> None:
        logger.debug(f"MLWorker started")

        # Load the model object
        try:
            model: BaseMLModel = self._load_model_func()
        except Exception as e:
            logger.error(
                f"MLWorker failed while loading the model. Error: {e}"
            )
            sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)
        logger.debug(f"MLWorker loaded the model")

        # Start processing messages using the loaded model and the scoring
        # function provided
        while True:
            message: JobMessage = self._message_queue.get()
            id_ = message.message_id
            args = message.args or []
            kwargs = message.kwargs or {}
            try:
                result = self._score_model_func(model, *args, **kwargs)
            except Exception as e:
                logger.error(
                    f"MLWorker failed while scoring model {model.__name__} "
                    f"with args {args}, kwargs {kwargs}. Error: {e}"
                )
                sys.exit(Config.USER_CODE_FAILED_EXIT_CODE)

            logger.debug(f"MLWorker successfully scored model")
            self._result_dict[id_] = result
