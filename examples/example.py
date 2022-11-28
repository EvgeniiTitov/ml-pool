from typing import Any
import time
import random
import threading

from ml_pool import MLPool
from ml_pool.utils import timer


# USER PROVIDED CODE
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


# IMITATING MULTIPLE CLIENTS / CALLERS


def call_pool(index: int) -> None:
    print(f"Called index {index} started")
    results = []
    for i in range(5):
        time.sleep(random.random())
        job_id = pool.schedule_model_scoring(f"{index}-{i}")
        result = pool.get_scoring_result(job_id, wait_if_not_available=True)
        results.append(result)

    print(f"Called index {index} done. Its results {results}")

    # TODO: Schedule multiple jobs, then get results


@timer
def main():
    # job_ids = []
    # for i in range(6):
    #     job_id = pool.schedule_model_scoring(i)
    #     job_ids.append(job_id)
    #
    # print("\n\n\nJob ids:", job_ids)
    #
    # for job_id in job_ids:
    #     result = pool.get_scoring_result(job_id)
    #     print(f"For id {job_id} result is {result}")

    caller_threads = [
        threading.Thread(target=call_pool, args=(i,)) for i in range(10)
    ]
    for caller_thread in caller_threads:
        caller_thread.start()

    for caller_thread in caller_threads:
        caller_thread.join()

    pool.shutdown()
    print("\nDone")


if __name__ == "__main__":
    pool = MLPool(load_model_func=load_model, score_model_func=score_model)
    main()
