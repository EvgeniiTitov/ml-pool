# flake8: noqa
import functools

import xgboost
import numpy as np
import pandas as pd
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from examples.models.train_text_classification import TextClassificationModel
from ml_pool.utils import timer
from ml_pool import MLPool


__all__ = ["TextClassifier", "HungryIris", "HungryDiabetesClassifier"]


class TextClassifier:
    def __init__(self, weights_path: str) -> None:
        self._model = TextClassificationModel(
            vocab_size=95811, embed_dim=64, num_class=4
        )
        self._model.load_state_dict(torch.load(weights_path))
        self._model.eval()
        self._model.to("cpu")  # Testing on Mac

        self._classes = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
        self._tokenizer = get_tokenizer("basic_english")

        train_iter = AG_NEWS(split="train")

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield self._tokenizer(text)

        self._vocab = build_vocab_from_iterator(
            yield_tokens(train_iter), specials=["<unk>"]
        )
        self._vocab.set_default_index(self._vocab["<unk>"])

        print("TextClassifier initialised")

    @timer
    def classify_text(self, text: str) -> str:
        with torch.no_grad():
            text = torch.tensor(self._preprocess_text(text))
            output = self._model(text, torch.tensor([0]))
            return self._classes[output.argmax(1).item() + 1]

    def _preprocess_text(self, text: str):
        return self._vocab(self._tokenizer(text))


class HungryModel:
    @staticmethod
    def simulate_cpu_load() -> None:
        summed = 0
        for i in range(5_000_000):
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


def load_text_classifier(weights_path: str) -> TextClassifier:
    return TextClassifier(weights_path)


def score_text_classifier(model: TextClassifier, text):
    return model.classify_text(text)


if __name__ == "__main__":
    # iris = HungryIris("iris_xgb.json")
    # start = time.perf_counter()
    # print(iris.predict([6.2, 2.2, 4.5, 1.5]))
    # print("Took:", time.perf_counter() - start)

    # diabetes = HungryDiabetesClassifier("diabetes_xgb.json")
    # start = time.perf_counter()
    # print(diabetes.predict([3, 187, 70, 22, 200, 36.4, 0.408, 36]))
    # print("Took:", time.perf_counter() - start)

    # text_classifier = TextClassifier("text_classification.pt")
    # for i in range(10):
    #     print(text_classifier.classify_text(
    #         text="MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    #         enduring the season’s worst weather conditions on Sunday at The \
    #         Open on his way to a closing 75 at Royal Portrush, which \
    #         considering the wind and the rain was a respectable showing. \
    #         Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    #         was another story. With temperatures in the mid-80s and hardly any \
    #         wind, the Spaniard was 13 strokes better in a flawless round. \
    #         Thanks to his best putting performance on the PGA Tour, Rahm \
    #         finished with an 8-under 62 for a three-stroke lead, which \
    #         was even more impressive considering he’d never played the \
    #         front nine at TPC Southwind."
    #     ))

    with MLPool(
        models_to_load={
            "text_classifier": functools.partial(
                load_text_classifier, "text_classification.pt"
            )
        },
        nb_workers=3,
    ) as pool:
        job_id = pool.create_job(
            score_model_function=score_text_classifier,
            model_name="text_classifier",
            kwargs={
                "text": "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
            enduring the season’s worst weather conditions on Sunday at The \
            Open on his way to a closing 75 at Royal Portrush, which \
            considering the wind and the rain was a respectable showing. \
            Thursday’s first round at the WGC-FedEx St. Jude Invitational \
            was another story. With temperatures in the mid-80s and hardly any \
            wind, the Spaniard was 13 strokes better in a flawless round. \
            Thanks to his best putting performance on the PGA Tour, Rahm \
            finished with an 8-under 62 for a three-stroke lead, which \
            was even more impressive considering he’d never played the \
            front nine at TPC Southwind."
            },
        )
        print(pool.get_result(job_id))
