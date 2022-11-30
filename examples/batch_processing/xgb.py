import xgboost
import time
import numpy as np
from functools import partial

from ml_pool.utils import timer
from ml_pool import MLPool


"""
Sync scoring takes: 15 sec

Scoring on the pool: 3.84 sec (4 workers), 1.83 sec (10 workers)
"""


def load_model(model_path: str):
    model = xgboost.Booster()
    model.load_model(model_path)
    return model


def score_model(model, features):
    # Imitates a heavy model that takes time to score + feature engineering
    # could also be unloaded to the worker pool
    time.sleep(0.15)

    features = xgboost.DMatrix([features])
    return np.argmax(model.predict(features))


def sync_scoring():
    model = load_model("../apis/iris_xgb.json")
    print("Model loaded")

    start = time.perf_counter()
    for _ in range(100):
        result = score_model(model, [6.2, 2.2, 4.5, 1.5])
        print(result)
    print(f"Took {time.perf_counter() - start: .4f}")

    print("Done")


@timer
def pool_scoring():
    job_ids = []
    for _ in range(100):
        job_ids.append(
            pool.schedule_scoring(kwargs={"features": [6.2, 2.2, 4.5, 1.5]})
        )

    for job_id in job_ids:
        print(pool.get_result(job_id, wait_if_unavailable=True))

    print("Done")


if __name__ == "__main__":
    with MLPool(
        load_model_func=partial(load_model, "iris_xgb.json"),
        score_model_func=score_model,
        nb_workers=10,
    ) as pool:
        pool_scoring()

    # sync_scoring()
