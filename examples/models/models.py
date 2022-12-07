import time

import xgboost
import numpy as np
import pandas as pd


__all__ = ["HungryIris", "HungryDiabetesClassifier"]


class HungryModel:
    @staticmethod
    def simulate_cpu_load() -> None:
        # Adds ~ 1 sec on my Mac
        summed = 0
        for i in range(20_000_000):
            summed += 1


class HungryIris(HungryModel):
    def __init__(self, weights_path: str) -> None:
        self._model = xgboost.Booster()
        self._model.load_model(weights_path)
        print("HungryIris initialised")

    def predict(self, features: list[float]):
        self.simulate_cpu_load()
        features = xgboost.DMatrix([features])
        return np.argmax(self._model.predict(features)[0])  # Batch size 1


class HungryDiabetesClassifier(HungryModel):
    def __init__(self, weights_path: str) -> None:
        self._model = xgboost.XGBClassifier()
        self._model.load_model(weights_path)
        print("HungryDiabetes initialised")

    def predict(self, features: list[float]):
        self.simulate_cpu_load()
        return self._model.predict(pd.DataFrame([features]))[0]  # Batch size 1


if __name__ == "__main__":
    # iris = HungryIris("iris_xgb.json")
    # start = time.perf_counter()
    # print(iris.predict([6.2, 2.2, 4.5, 1.5]))
    # print("Took:", time.perf_counter() - start)

    diabetes = HungryDiabetesClassifier("diabetes_xgb.json")
    start = time.perf_counter()
    print(diabetes.predict([3, 187, 70, 22, 200, 36.4, 0.408, 36]))
    print("Took:", time.perf_counter() - start)
