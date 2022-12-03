import time
from functools import partial

import pytest

from ml_pool import MLPool
from ml_pool.exceptions import (
    UserProvidedCallableError,
    JobWithSuchIDDoesntExistError,
)
from ml_pool.utils import get_new_job_id
from ml_pool.config import Config


# ---------------------------------- models ----------------------------------
class GoodModel1:
    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath
        print(f"GoodModel {name} loaded from {filepath}")

    def predict(self, *args, **kwargs):
        time.sleep(0.2)
        return args


class GoodModel2:
    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath
        print(f"GoodModel {name} loaded from {filepath}")

    def predict(self, *args, **kwargs):
        time.sleep(0.2)
        return kwargs


class BadModel:
    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath
        print(f"BadModel {name} loaded from {filepath}")

    def predict(self, *args, **kwargs):
        raise Exception("Failed")


# ---------------------------- loading functions -----------------------------


def load_good_model_one(filepath: str) -> GoodModel1:
    return GoodModel1("one", filepath)


def load_good_model_two(filepath: str) -> GoodModel2:
    return GoodModel2("two", filepath)


def load_bad_model_one(filepath: str) -> BadModel:
    return BadModel("one", filepath)


def load_bad_model_two(filepath: str) -> BadModel:
    return BadModel("two", filepath)


def faulty_load_model(filepath: str) -> GoodModel1:
    raise Exception("Failed")


# ---------------------------- scoring functions -----------------------------


def score_model(model, *args, **kwargs):
    return model.predict(*args, **kwargs)


def faulty_score_model(model, *args, **kwargs):
    raise Exception("Failed")


# ---------------------------------- tests -----------------------------------


def test_good_load_good_model():
    with MLPool(
        models_to_load={
            "return_args_model": partial(load_good_model_one, "filepath_1"),
            "return_kwargs_model": partial(load_good_model_two, "filepath_2"),
        },
        nb_workers=1,
    ) as _:
        pass


def test_bad_load_good_model():
    with pytest.raises(UserProvidedCallableError):
        with MLPool(
            models_to_load={
                "return_args_model": "definitely_not_callable",
                "return_kwargs_model": partial(
                    load_good_model_two, "filepath_2"
                ),
            },
            nb_workers=1,
        ) as _:
            pass


def test_pool_shutdown():
    pool = MLPool(
        models_to_load={
            "return_args_model": partial(load_good_model_one, "filepath_1"),
            "return_kwargs_model": partial(load_good_model_two, "filepath_2"),
        },
        nb_workers=1,
    )
    pool.shutdown()
    assert pool._pool_running is False


def test_pool_single_task():
    with MLPool(
        models_to_load={
            "return_args_model": partial(load_good_model_one, "filepath_1"),
            "return_kwargs_model": partial(load_good_model_two, "filepath_2"),
        },
        nb_workers=1,
    ) as pool:
        job_id = pool.create_job(
            score_model_function=score_model,
            model_name="return_args_model",
            args=(1, 2, 3),
            kwargs={"four": 4},
        )
        result = pool.get_result(job_id)
        assert result == (1, 2, 3)


def test_pool_multiple_tasks():
    with MLPool(
        models_to_load={
            "return_args_model": partial(load_good_model_one, "filepath_1"),
            "return_kwargs_model": partial(load_good_model_two, "filepath_2"),
        },
        nb_workers=3,
    ) as pool:
        job_id_1 = pool.create_job(
            score_model, "return_args_model", (1, 2, 3), {"four": 4}
        )
        job_id_2 = pool.create_job(
            score_model, "return_kwargs_model", (1, 2, 3), {"four": 4}
        )
        ids = [
            pool.create_job(score_model, "return_args_model", args=(i,))
            for i in range(5)
        ]
        results = [pool.get_result(id) for id in ids]

        assert pool.get_result(job_id_1) == (1, 2, 3)
        assert pool.get_result(job_id_2) == {"four": 4}
        assert results == [(0,), (1,), (2,), (3,), (4,)]


def test_pool_unknown_model_for_scoring():
    with pytest.raises(ValueError):
        with MLPool(
            models_to_load={
                "return_args_model": partial(
                    load_good_model_one, "filepath_1"
                ),
                "return_kwargs_model": partial(
                    load_good_model_two, "filepath_2"
                ),
            },
            nb_workers=1,
        ) as pool:
            job_id = pool.create_job(
                score_model_function=score_model,
                model_name="idk_this_model",
                args=(1, 2, 3),
                kwargs={"four": 4},
            )
            result = pool.get_result(job_id)
            assert result == (1, 2, 3)


def test_pool_request_result_for_unknown_id():
    with pytest.raises(JobWithSuchIDDoesntExistError):
        with MLPool(
            models_to_load={
                "return_args_model": partial(
                    load_good_model_one, "filepath_1"
                ),
                "return_kwargs_model": partial(
                    load_good_model_two, "filepath_2"
                ),
            },
            nb_workers=1,
        ) as pool:
            _ = pool.create_job(
                score_model_function=score_model,
                model_name="return_args_model",
                args=(1, 2, 3),
                kwargs={"four": 4},
            )
            result = pool.get_result(get_new_job_id())
            assert result == (1, 2, 3)


def heat_death_score(model, *args, **kwargs):
    time.sleep(10000)


def test_pool_avoid_blocking_when_creating_new_job():
    with MLPool(
        models_to_load={
            "return_args_model": partial(load_good_model_one, "filepath_1"),
            "return_kwargs_model": partial(load_good_model_two, "filepath_2"),
        },
        nb_workers=1,
        message_queue_size=50,  # Min size allowed
    ) as pool:
        for i in range(51):
            pool.create_job(heat_death_score, "return_args_model")

        job_id = pool.create_job(
            score_model, "return_args_model", wait_if_full=False
        )
        assert job_id is None


def test_pool_avoid_blocking_when_result_not_ready():
    with MLPool(
        models_to_load={
            "return_args_model": partial(load_good_model_one, "filepath_1"),
            "return_kwargs_model": partial(load_good_model_two, "filepath_2"),
        },
        nb_workers=1,
    ) as pool:
        job_id = pool.create_job(heat_death_score, "return_args_model")
        result = pool.get_result(job_id, wait_if_unavailable=False)
        assert result is None


def slowish_score(model, *args, **kwargs):
    time.sleep(3.0)


def test_pool_cancel_task():
    with MLPool(
        models_to_load={
            "return_kwargs_model": partial(load_good_model_two, "filepath_2")
        },
        nb_workers=1,
    ) as pool:
        _ = pool.create_job(slowish_score, "return_kwargs_model", args=("1",))

        job_id_2 = pool.create_job(score_model, "return_kwargs_model")
        pool.cancel_job(job_id_2)

        time.sleep(1.0)
        assert job_id_2 not in pool._scheduled_job_ids
        assert job_id_2 not in pool._result_dict
        assert job_id_2 in pool._cancel_dict


def test_pool_expired_results_cleaning():
    original_ttl = Config.RESULT_TTL
    original_cleaning_freq = Config.CLEANER_THREAD_FREQUENCY

    Config.RESULT_TTL = 1.0
    Config.CLEANER_THREAD_FREQUENCY = 0.2

    with MLPool(
        models_to_load={
            "return_kwargs_model": partial(load_good_model_two, "filepath_2")
        },
        nb_workers=1,
        result_ttl=1.0,
    ) as pool:
        job_id_2 = pool.create_job(score_model, "return_kwargs_model")
        time.sleep(2.0)

        with pytest.raises(JobWithSuchIDDoesntExistError):
            _ = pool.get_result(job_id_2)

    Config.RESULT_TTL = original_ttl  # Not sure if it can affect other tests
    Config.CLEANER_THREAD_FREQUENCY = original_cleaning_freq


def test_pool_monitorigng_thread_restarts_failed_workers():
    total_workers = 3
    with MLPool(
        models_to_load={
            "return_kwargs_model": partial(load_good_model_two, "filepath_2")
        },
        nb_workers=total_workers,
    ) as pool:
        pool._workers[1].terminate()
        time.sleep(1.0)
        assert len(pool._workers) == total_workers

        for worker in pool._workers:
            worker.terminate()

        time.sleep(1.0)
        assert len(pool._workers) == total_workers


# TODO: Test exit codes
# Test pool bad score callable (throws exception)
# etc
