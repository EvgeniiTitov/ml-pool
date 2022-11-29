import pytest

from ml_pool import MLPool
from ml_pool.exceptions import UserProvidedCallableFailedError


class Model:
    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        return args, kwargs


def load_model():
    return Model()


def score_model(model, *args, **kwargs):
    return model.predict(*args, **kwargs)


def test_providing_faulty_load_model_callable():
    with pytest.raises(UserProvidedCallableFailedError):
        _ = MLPool("absolutely_not_callable", score_model, 1)


def test_providing_faulty_score_model_callable():
    with pytest.raises(UserProvidedCallableFailedError):
        _ = MLPool(load_model, "absolutely_not_callable", 1)


def test_pool():
    with MLPool(load_model, score_model, 1) as pool:
        job_id = pool.schedule_model_scoring(1, test=2)
        result = pool.get_scoring_result(job_id, wait_if_not_available=True)
        assert result == ((1,), {"test": 2})


def test_pool_shutdown():
    pool = MLPool(load_model, score_model, 1)
    pool.shutdown()
    assert pool._pool_running is False
