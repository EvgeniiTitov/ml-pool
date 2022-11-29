import multiprocessing
import time

import pytest

from ml_pool.worker import MLWorker
from ml_pool.config import Config
from ml_pool.utils import get_new_job_id
from ml_pool.messages import JobMessage


@pytest.fixture(scope="module")
def manager():
    return multiprocessing.Manager()


@pytest.fixture(scope="module")
def result_dict(manager):
    return manager.dict()


@pytest.fixture()
def queue():
    return multiprocessing.Queue()


class Model:
    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        return args, kwargs


def load_model():
    return Model()


def bad_load_model():
    raise Exception("Failed")


def score_model(model, *args, **kwargs):
    return model.predict(*args, **kwargs)


def bad_score_model():
    raise Exception("Failed")


def test_providing_faulty_load_model_callable(result_dict, queue):
    worker = MLWorker(result_dict, queue, bad_load_model, score_model)
    worker.start()

    time.sleep(1.0)

    assert not worker.is_alive()
    assert worker.exitcode == Config.USER_CODE_FAILED_EXIT_CODE

    worker.terminate()
    worker.join()


def test_providing_faulty_score_model_callable(result_dict, queue):
    worker = MLWorker(result_dict, queue, load_model, bad_score_model)
    worker.start()

    queue.put(JobMessage(message_id=get_new_job_id(), args=[1]))

    time.sleep(1.0)

    assert not worker.is_alive()
    assert worker.exitcode == Config.USER_CODE_FAILED_EXIT_CODE

    worker.terminate()
    worker.join()


def test_worker(result_dict, queue):
    message_id = get_new_job_id()

    worker = MLWorker(result_dict, queue, load_model, score_model)
    worker.start()

    queue.put(JobMessage(message_id=message_id, args=[2], kwargs={"test": 3}))

    time.sleep(1.0)

    assert message_id in result_dict
    assert result_dict[message_id] == ((2,), {"test": 3})

    worker.terminate()
    worker.join()
