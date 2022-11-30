import time

import pytest
import uuid

from ml_pool import MLPool
from ml_pool.exceptions import (
    UserProvidedCallableFailedError,
    JobWithSuchIDDoesntExistError,
)
from ml_pool.utils import get_new_job_id
from ml_pool.config import Config


class Model:
    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        return args, kwargs


def load_model():
    return Model()


def score_model(model, *args, **kwargs):
    return model.predict(*args, **kwargs)


def faulty_score_model(model, *args, **kwargs):
    raise Exception("Failed")


def test_providing_faulty_load_model_callable():
    with pytest.raises(UserProvidedCallableFailedError):
        _ = MLPool("absolutely_not_callable", score_model, 1)


def test_providing_faulty_score_model_callable():
    with pytest.raises(UserProvidedCallableFailedError):
        _ = MLPool(load_model, "absolutely_not_callable", 1)


def test_pool_single_task():
    with MLPool(load_model, score_model, 1) as pool:
        job_id = pool.schedule_scoring(args=(1,), kwargs={"test": 2})
        result = pool.get_result(job_id, wait_if_unavailable=True)
        assert result == ((1,), {"test": 2})


def test_pool_multiple_tasks():
    with MLPool(load_model, score_model, 2) as pool:
        job_ids = [pool.schedule_scoring(args=(i,)) for i in range(5)]
        results = [
            pool.get_result(job_id, wait_if_unavailable=True)
            for job_id in job_ids
        ]
        assert results == [
            ((0,), {}),
            ((1,), {}),
            ((2,), {}),
            ((3,), {}),
            ((4,), {}),
        ]


def test_getting_results_for_unknown_id():
    with pytest.raises(JobWithSuchIDDoesntExistError):
        with MLPool(load_model, score_model, 1) as pool:
            _ = pool.get_result(
                job_id=get_new_job_id(), wait_if_unavailable=True
            )


def blocking_scoring(model, *args, **kwargs):
    import time

    time.sleep(10000)


def test_pools_scheduling():
    with MLPool(
        load_model,
        blocking_scoring,
        nb_workers=1,
        message_queue_size=Config.DEFAULT_MIN_QUEUE_SIZE,
    ) as pool:
        job_id = pool.schedule_scoring(args=(1,))
        assert isinstance(job_id, uuid.UUID)

        # Min inner queue size is 50, fill the queue
        for _ in range(Config.DEFAULT_MIN_QUEUE_SIZE):
            pool.schedule_scoring(args=(1,))

        # Make sure the next call doesn't block the caller
        supposed_none = pool.schedule_scoring(
            args=(1,), block_until_scheduled=False
        )
        assert supposed_none is None


def test_pool_shutdown():
    pool = MLPool(load_model, score_model, 1)
    assert pool._pool_running is True
    pool.shutdown()
    assert pool._pool_running is False


def test_pool_closes_with_faulty_user_code():
    with pytest.raises(UserProvidedCallableFailedError):
        with MLPool(load_model, faulty_score_model, 1) as pool:
            task_id = pool.schedule_scoring(args=(1,))
            time.sleep(1.1)
            assert pool._workers_healthy is False
            assert pool._workers_exception is not None
            _ = pool.get_result(task_id, wait_if_unavailable=True)
