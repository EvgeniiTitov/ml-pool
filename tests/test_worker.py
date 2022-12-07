import multiprocessing
from functools import partial
import time

import pytest

from ml_pool.worker import MLWorker
from ml_pool.utils import get_new_job_id
from ml_pool.messages import JobMessage
from ml_pool.config import Config


@pytest.fixture(scope="function")
def manager():
    manager = multiprocessing.Manager()
    yield manager
    manager.shutdown()


@pytest.fixture(scope="function")
def queue():
    return multiprocessing.Queue()


@pytest.fixture(scope="function")
def result_dict(manager):
    return manager.dict()


@pytest.fixture(scope="function")
def cancel_dict(manager):
    return manager.dict()


class Model:
    def __init__(self, name: str):
        self.name = name

    def predict(self, *args, **kwargs):
        return args, kwargs


def good_load_model(name: str) -> Model:
    return Model(name)


def bad_load_model(name: str) -> Model:
    raise Exception("Failed")


def bad_load_returns_nothing():
    return


def good_score_model(model, *args, **kwargs):
    return model.predict(*args, **kwargs)


def good_score_model_slow(model, *args, **kwargs):
    time.sleep(1.0)
    return model.predict(*args, **kwargs)


def bad_score_model(model, *args, **kwargs):
    raise Exception("Failed")


# ----------------------------------- tests -----------------------------------


def test_worker_good_load_good_score(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={"model_1": partial(good_load_model, "name1")},
    )
    worker.start()

    assert "model_1" in worker._ml_models

    job_id = get_new_job_id()
    queue.put(
        JobMessage(job_id, good_score_model, "model_1", (1,), {"test": 1})
    )

    time.sleep(2.0)  # Assumed

    assert job_id in result_dict
    assert result_dict[job_id][1] == ((1,), {"test": 1})

    worker.terminate()
    worker.join()


def test_worker_good_load_good_score_multiple(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={
            "model_1": partial(good_load_model, "name1"),
            "model_2": partial(good_load_model, "name2"),
        },
    )
    worker.start()

    job_ids = []
    for i in range(10):
        job_id = get_new_job_id()
        queue.put(
            JobMessage(job_id, good_score_model, "model_1", (i,), {"test": i})
        )
        job_ids.append(job_id)

    time.sleep(5.0)  # Assumed

    for i, job_id in enumerate(job_ids):
        assert job_id in result_dict
        assert result_dict[job_id][1] == ((i,), {"test": i})

    worker.terminate()
    worker.join()


def test_worker_good_load_bad_score(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={"model_1": partial(good_load_model, "name1")},
    )
    worker.start()

    job_id = get_new_job_id()
    queue.put(JobMessage(job_id, bad_score_model, "model_1", (1,)))

    time.sleep(2.0)

    assert job_id not in result_dict
    assert worker.exitcode == Config.SCORE_MODEL_CALLABLE_FAILED
    assert not worker.is_alive()

    worker.terminate()
    worker.join()


def test_worker_bad_load(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={"model_1": partial(bad_load_model, "name1")},
    )
    worker.start()

    time.sleep(2.0)

    assert not worker.is_alive()
    assert worker.exitcode == Config.LOAD_MODEL_CALLABLE_FAILED

    worker.terminate()
    worker.join()


def test_worker_bad_load_returns_nothing(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={"model_1": bad_load_returns_nothing},
    )
    worker.start()

    time.sleep(2.0)

    assert not worker.is_alive()
    assert worker.exitcode == Config.LOAD_MODEL_CALLABLE_RETURNED_NOTHING

    worker.terminate()
    worker.join()


def test_worker_cancelling_job(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={"model_1": partial(good_load_model, "name1")},
    )
    worker.start()

    job_id = get_new_job_id()
    queue.put(
        JobMessage(
            job_id, good_score_model_slow, "model_1", kwargs={"test": 1}
        )
    )
    job_id_2 = get_new_job_id()
    queue.put(
        JobMessage(
            job_id_2, good_score_model_slow, "model_1", kwargs={"test": 1}
        )
    )
    cancel_dict[job_id_2] = None

    time.sleep(3.0)
    assert job_id_2 not in result_dict
    assert job_id_2 not in cancel_dict

    worker.terminate()
    worker.join()


def test_worker_graceful_stopping(queue, result_dict, cancel_dict):
    worker = MLWorker(
        queue,
        result_dict,
        cancel_dict,
        ml_models={"model_1": partial(good_load_model, "name1")},
    )
    worker.start()
    worker.initiate_stop()

    time.sleep(1.0)
    assert not worker.is_alive()
    assert worker.exitcode == 0

    worker.join()
