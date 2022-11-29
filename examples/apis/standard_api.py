import sys

sys.path.append("../..")

from fastapi import FastAPI
import pydantic
import xgboost
import numpy as np
import uvicorn

from ml_pool.logger import get_logger


logger = get_logger("api")

app = FastAPI()


def load_model(model_path: str):
    model = xgboost.Booster()
    model.load_model(model_path)
    logger.info("Model loaded")
    return model


model = load_model("iris_xgb.json")


def score_model(model, features):
    # Imitates a heavy model that takes time to score + feature engineering
    # could also be unloaded to the worker pool
    sum_ = 0
    for i in range(10_000_000):
        sum_ += 1

    features = xgboost.DMatrix([features])
    return np.argmax(model.predict(features))


class Request(pydantic.BaseModel):
    features: list[float]


class Response(pydantic.BaseModel):
    prediction: int


@app.get("/")
def health_check():
    return {"Message": "Up and running"}


@app.post("/iris")
def score(request: Request) -> Response:
    logger.info(f"Got request for features: {request}")
    result = score_model(model, request.features)
    return Response(prediction=result)


if __name__ == "__main__":
    uvicorn.run(app, workers=1)
