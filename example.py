from typing import Any
import time

from ml_pool import MLPool, BaseMLModel


class Model(BaseMLModel):

    def score(self, features) -> Any:
        pass


def load_model() -> BaseMLModel:
    return Model()


def score_model(model: BaseMLModel, features):
    return model.score(features)


def main():
    time.sleep(60)
    pool.shutdown()


if __name__ == '__main__':
    pool = MLPool(load_model_func=load_model, score_model_func=score_model)
    main()
