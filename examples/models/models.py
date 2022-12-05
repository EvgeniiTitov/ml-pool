import time

import xgboost
import numpy as np


class HungryIris:
    def __init__(self, weights_path: str) -> None:
        self._model = xgboost.Booster()
        self._model.load_model(weights_path)
        print("SlowishIris initialised")

    def predict(self, features: list[float]):
        self.simulate_cpu_load()
        features = xgboost.DMatrix([features])
        return np.argmax(self._model.predict(features)[0])  # Batch size 1

    def simulate_cpu_load(self) -> None:
        # Adds ~ 1 sec on my Mac
        summed = 0
        for i in range(20_000_000):
            summed += 1


if __name__ == "__main__":
    iris = HungryIris("iris_xgb.json")
    start = time.perf_counter()
    print(iris.predict([6.2, 2.2, 4.5, 1.5]))
    print("Took:", time.perf_counter() - start)
