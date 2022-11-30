import torch
from functools import partial

from ml_pool import MLPool
from ml_pool.utils import timer


"""
Sync - 13.3 sec sync scoring 100 times
On the pool - TBA (cant load model, the well known torch problem...)
"""


def load_model(repo: str, model_name: str):
    import sys

    sys.path.insert(0, "../yolov5s.pt")
    return torch.hub.load(repo_or_dir=repo, model=model_name)


def score_model(model, image_path: str):
    return model(image_path)


@timer
def score_sync():
    model = load_model(repo="ultralytics/yolov5", model_name="yolov5s")
    print("Model loaded")

    for _ in range(100):
        preds = score_model(model, "/Users/etitov1/Downloads/zidane.jpeg")
        preds.print()

    print("Done")


@timer
def score_on_the_pool():
    ids = []
    for _ in range(100):
        id_ = pool.schedule_scoring(
            kwargs={"image_path": "/Users/etitov1/Downloads/zidane.jpeg"}
        )
        ids.append(id_)

    for id_ in ids:
        result = pool.get_result(id_, wait_if_unavailable=True)
        print(result)


if __name__ == "__main__":
    pool = MLPool(
        load_model_func=partial(load_model, "ultralytics/yolov5", "yolov5s"),
        score_model_func=score_model,
    )
    score_on_the_pool()

    # score_sync()
