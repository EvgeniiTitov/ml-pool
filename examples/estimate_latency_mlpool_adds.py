import sys

sys.path.append("..")
from functools import partial
import time

from ml_pool import MLPool
from examples.models import HungryIris


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


def main():
    score_directly()


if __name__ == "__main__":
    with MLPool(
        models_to_load={"iris": partial(load_iris, "./models/iris_xgb.json")},
        nb_workers=1,
    ) as pool:
        score_on_pool()

    score_directly()
