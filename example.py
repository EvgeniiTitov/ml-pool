from typing import Any
import time

from ml_pool import MLPool


class Model:
    def __init__(self):
        pass

    def score(self, features) -> Any:
        print("Scoring model for features:", features)
        time.sleep(1.0)


def load_model():
    return Model()


def score_model(model, features):
    return model.score(features)


def main():
    time.sleep(60)
    pool.shutdown()


if __name__ == "__main__":
    pool = MLPool(load_model_func=load_model, score_model_func=score_model)
    main()
