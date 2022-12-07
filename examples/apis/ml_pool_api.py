import sys

sys.path.append("../..")

from fastapi import FastAPI
import pydantic
import uvicorn

from ml_pool import MLPool
from ml_pool.logger import get_logger
from examples.models import HungryIris


logger = get_logger("api")

app = FastAPI()


# --------------------- functions a user to provide --------------------------


def load_model() -> HungryIris:
    return HungryIris("../models/iris_xgb.json")


def score_model(model: HungryIris, features):
    return model.predict(features)


# ------------------------------- schemas ------------------------------------


class Request(pydantic.BaseModel):
    features: list[float]


class Response(pydantic.BaseModel):
    prediction: int


# ------------------------------- endpoints ----------------------------------


@app.get("/")
def health_check():
    return {"Message": "Up and running"}


# Checking result cleaning thread
@app.post("/create_task")
def create_task(request: Request):
    logger.info(f"Got request for features: {request}")
    pool.create_job(
        score_model_function=score_model,
        model_name="hungry_iris",
        kwargs={"features": request.features},
    )


@app.post("/iris")
def score(request: Request) -> Response:
    logger.info(f"Got request for features: {request}")

    # UNLOAD DATA CRUNCHING CPU HEAVY MODEL SCORING TO THE POOL WITHOUT
    # OVERLOADING THE API PROCESS
    job_id = pool.create_job(
        score_model_function=score_model,
        model_name="hungry_iris",
        kwargs={"features": request.features},
    )
    result = pool.get_result(job_id)
    return Response(prediction=result)


if __name__ == "__main__":
    with MLPool(models_to_load={"hungry_iris": load_model}) as pool:
        uvicorn.run(app, workers=1)
