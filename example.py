from typing import Any
import time

from ml_pool import MLPool
from ml_pool.utils import timer


class Model:
    def __init__(self):
        pass

    def predict(self, features) -> Any:
        print("Scoring model for features:", features)
        time.sleep(2.0)
        return features


def load_model():
    return Model()


def score_model(model, features):
    return model.predict(features)


@timer
def main():
    job_ids = []
    for i in range(6):
        job_id = pool.schedule_model_scoring(i)
        job_ids.append(job_id)

    print("\n\n\nJob ids:", job_ids)

    for job_id in job_ids:
        result = pool.get_scoring_result(job_id)
        print(f"For id {job_id} result is {result}")

    pool.shutdown()


if __name__ == "__main__":
    pool = MLPool(load_model_func=load_model, score_model_func=score_model)
    main()
