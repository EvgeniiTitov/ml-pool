import sys

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
def load_iris() -> HungryIris:
    return HungryIris("../models/iris_xgb.json")


def score_iris(model: HungryIris, features):
    return model.predict(features)


def load_diabetes_classifier():
    return HungryDiabetesClassifier


def score_diabetes_classifier():
    pass


# ------------------------------- schemas ------------------------------------
class Request(pydantic.BaseModel):
    features: list[float]


class Response(pydantic.BaseModel):
    prediction: int


# ------------------------------- endpoints ----------------------------------
@app.post("/iris")
async def serve_iris(request: Request) -> Response:
    pass


@app.post("/diabetes")
async def serve_diabetes_classifier(request: Request) -> Response:
    pass


if __name__ == "__main__":
    with MLPool(
        models_to_load={
            "iris": load_iris,
            "diabetes_classifier": load_diabetes_classifier,
        }
    ) as pool:
        uvicorn.run(app, workers=1)
