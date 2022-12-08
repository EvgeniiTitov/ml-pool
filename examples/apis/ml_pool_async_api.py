import sys
from functools import partial

sys.path.append("../..")

from fastapi import FastAPI
import pydantic
import uvicorn

from ml_pool import MLPool
from ml_pool.logger import get_logger
from examples.models import HungryIris, HungryDiabetesClassifier


logger = get_logger("api")

app = FastAPI()


# --------------------- functions a user to provide --------------------------
def load_iris(model_path: str) -> HungryIris:
    # TODO: Loading and preparing ML model goes here

    return HungryIris(model_path)


def score_iris(model: HungryIris, features):
    # TODO: Feature engineering etc goes here

    return model.predict(features)


def load_diabetes_classifier(model_path: str) -> HungryDiabetesClassifier:
    # TODO: Loading and preparing ML model goes here

    return HungryDiabetesClassifier(model_path)


def score_diabetes_classifier(model: HungryDiabetesClassifier, features):
    # TODO: Feature engineering etc goes here

    return model.predict(features)


# ------------------------------- schemas ------------------------------------
class Request(pydantic.BaseModel):
    features: list[float]


class Response(pydantic.BaseModel):
    prediction: int


# ------------------------------- endpoints ----------------------------------
@app.post("/iris")
async def serve_iris(request: Request) -> Response:
    features = request.features
    logger.info(f"/iris request, features: {features}")

    job_id = await pool.create_job_async(
        score_model_function=score_iris,
        model_name="iris",
        kwargs={"features": features},
    )
    result = await pool.get_result_async(job_id)

    return Response(prediction=result)


@app.post("/diabetes")
async def serve_diabetes_classifier(request: Request) -> Response:
    features = request.features
    logger.info(f"/diabetes request, features: {features}")

    job_id = await pool.create_job_async(
        score_model_function=score_diabetes_classifier,
        model_name="diabetes_classifier",
        args=(features,),
    )
    result = await pool.get_result_async(job_id)

    return Response(prediction=result)


if __name__ == "__main__":
    with MLPool(
        models_to_load={
            "iris": partial(load_iris, "../models/iris_xgb.json"),
            "diabetes_classifier": partial(
                load_diabetes_classifier, "../models/diabetes_xgb.json"
            ),
        },
        nb_workers=5,
    ) as pool:
        uvicorn.run(app, workers=1)
