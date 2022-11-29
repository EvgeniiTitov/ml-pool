import sys

sys.path.append("../..")

from functools import partial

from fastapi import FastAPI
import pydantic
import xgboost
import numpy as np
import uvicorn

from ml_pool import MLPool
from ml_pool.logger import get_logger


logger = get_logger("api")

app = FastAPI()


# ------------------ user provided callables to load and score ---------------


def load_model(model_path: str):
    model = xgboost.Booster()
    model.load_model(model_path)
    return model


def score_model(model, features):
    # Imitates a heavy model that takes time to score + feature engineering
    # could also be unloaded to the worker pool
    sum_ = 0
    for i in range(10_000_000):
        sum_ += 1

    features = xgboost.DMatrix([features])
    return np.argmax(model.predict(features))


# ------------------------------- schemas ------------------------------------


class Request(pydantic.BaseModel):
    features: list[float]


class Response(pydantic.BaseModel):
    prediction: int


# ------------------------------- endpoints ----------------------------------


@app.get("/")
def health_check():
    return {"Message": "Up and running"}


@app.post("/iris")
def score(request: Request) -> Response:
    logger.info(f"Got request for features: {request}")

    # UNLOAD DATA CRUNCHING CPU HEAVY MODEL SCORING TO THE POOL WITHOUT
    # OVERLOADING THE API PROCESS
    job_id = pool.schedule_model_scoring(features=request.features)
    result = pool.get_scoring_result(job_id, wait_if_unavailable=True)

    return Response(prediction=result)


if __name__ == "__main__":
    with MLPool(
        load_model_func=partial(load_model, "iris_xgb.json"),
        score_model_func=score_model,
    ) as pool:
        uvicorn.run(app, workers=1)
