import sys
import asyncio

sys.path.append("..")
from functools import partial
import time

from ml_pool import MLPool
from examples.models import HungryIris


"""
The point is to estimate how much latency MLPool adds compared to running the
model directly.

Under the hood MLPool does a bunch of things:
- needs to do some checks
- create a Job object
- put Job in the queue
- then, MLWorker gets it from the queue
- processes the job,
- puts in the shared dictionary

All of these ^ takes time, but how much?
"""


NB_SCORES = 100


def load_iris(model_path: str) -> HungryIris:
    return HungryIris(model_path)


def score_iris(model: HungryIris, features):
    return model.predict(features)


def score_directly():
    model = load_iris("./models/iris_xgb.json")

    start = time.perf_counter()
    for i in range(NB_SCORES):
        _ = score_iris(model, features=[3.0, 2.0, 1.0, 0.2])
        print(f"Scored {i} time")

    print("Direct model scoring took:", time.perf_counter() - start)


def score_on_pool():
    start = time.perf_counter()
    for i in range(NB_SCORES):
        job_id = pool.create_job(
            score_iris, model_name="iris", args=([3.0, 2.0, 1.0, 0.2],)
        )
        _ = pool.get_result(job_id)
        print(f"Scored {i} time")

    print("Pool scoring took:", time.perf_counter() - start)


# Just to make sure
async def score_on_pool_async():
    start = time.perf_counter()
    for i in range(NB_SCORES):
        job_id = await pool.create_job_async(
            score_iris, model_name="iris", args=([3.0, 2.0, 1.0, 0.2],)
        )
        _ = await pool.get_result_async(job_id)
        print(f"Scored {i} time")

    print("Async pool scoring took:", time.perf_counter() - start)


if __name__ == "__main__":
    with MLPool(
        models_to_load={"iris": partial(load_iris, "./models/iris_xgb.json")},
        nb_workers=1,
    ) as pool:
        score_on_pool()
        asyncio.run(score_on_pool_async())

    score_directly()
