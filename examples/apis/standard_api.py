import sys

sys.path.append("../..")

from fastapi import FastAPI
import pydantic
import uvicorn

from ml_pool.logger import get_logger
from examples.models import HungryIris


logger = get_logger("api")

app = FastAPI()

model = HungryIris("../models/iris_xgb.json")


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
    result = model.predict(request.features)
    return Response(prediction=result)


if __name__ == "__main__":
    uvicorn.run(app, workers=1)
