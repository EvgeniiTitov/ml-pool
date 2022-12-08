### Use case / why


Say, we have a ML model, which is CPU hungry and takes time to score, which we want to 
serve through an API / Web Sockets while:

- keeping overall application complexity low (no dedicated service to serve ML models)
- minimising latency
- maximasing throughput (the number of requests a single Cloud Run instance / container can handle)
- providing the ability to avoid making blocking calls, which would block coroutines / web sockets

A typical simple API serving ML model(s) instantiates the models right in the main process, which
results in the models consuming the API process's RAM and CPU cycles for scoring. This directly 
affects the API performance in terms of latency and throughput. It would be helpful if we could move 
model scoring bit somewhere else away from the API process - other cores. This would let us:

- use CPU resources more efficiently (loading more cores instead of a single one provided that the container has 1+ cores at its disposal)
- avoid hurting API performance by moving model scoring that require memory and CPU away from the API process
- decrease latency
- increase throughput 

On top of that, the solution seems to integrate well with the Cloud Run autoscaling strategy. When 
the load is low, the workers serving the models are pretty much idle. As the load increases,
the workers get busy, which in turn increases overall Cloud Run instance CPU usage. Once
it reaches a certain threshold, Cloud Run will spin up another instance to handle the load.


---

### How to use / Examples

- Instantiating MLPool

In order to use MLPool, for every ML model a user wants to run on the pool, they need to provide 
a callable, which loads a model into memory and prepares it. Under the hood, MLPool will ensure 
every worker in the pool running in a dedicated process loads the model(s) in memory when it starts (done only once).

Consider these functions that load the text classification models as an example:

```python
def load_text_classifier(weights_path: str) -> TextClassifier:
    return TextClassifier(weights_path)
```

When instantiating MLPool, a user needs to pass the parameter models_to_load, which is a dictionary where the keys are
model names MLPool will serve, and the values are the callables loading the models (example function above).

In addition, the user can specify the number of workers they want, scoring results TTL, the number of jobs
that could be queued on the pool etc.

```python
if __name__ == "__main__":
    with MLPool(
        models_to_load={
            "text_classifier": partial(load_text_classifier, "text_classification.pt"),
            "any_other_model": load_any_other_model_callable
        },
        nb_workers=5,
    ) as pool:
        ...
```

! IMPORTANT, when instantiating MLPool do it under `if __name__ == "__main__":`

- Scoring on MLPool

MLPool gives user full control of what gets executed on the pool, so in order to score a model
on the pool, the user needs to provide a callable such as:

```python
def score_text_classifier(model: TextClassifier, text):
    return model.classify_text(text)
```
where the first parameter is the model and then anything else the user wants to pass

To schedule execution of the function above on the pool, the user can call `.create_job()` or asyncio friendly
`.create_job_async()` methods passing the:

- `score_model_function` - a function to run on the pool, which accepts a model as the first argument
- `model_name` - model to be passed to the function as the first argument, must be one of the functions provided when instantiating MLPool
- `args`, `kwargs` - any other arguments to pass to the function

```python
job_id = pool.create_job(
    score_model_function=score_text_classifier,
    model_name="text_classifier",
    kwargs={
        "text": "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing"
    }
)
print(pool.get_result(job_id))
```

To execution results back the user can call `get_result` or asyncio friendly `get_result_async` passing
id of the job.


- Cancelling

Once a job was created, it is possible to cancel model scoring if it hasn't run yet using `.cancel_job()`. Convenient, if, 
say, your web socket client disconnects

---

### End to end example:

Say, we have two models we want to serve.

```python
from functools import partial

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

```

---

### Gotchas:

- MLPool comes with significant overhead which has to do with IPC in Python (pickling/unpicking, etc), 
so if your model is light and takes very little time to run (~milliseconds), then introducing MLPool will
only slow things down. In this case it might make sense to score the model directly in the API process. 
If there is heavy feature engineering work associated with scoring the model, then it could 
make sense, it depends.

- ! It is common to spin up multiple instances of the API app inside a container using tools such as
gunicorn etc. Be careful when using MLPool in such configuration as you could overload CPU constantly triggering
Cloud Run to scale up lol. In our case each container runs only a single instance of the API app, spinning up more instances
within the same container won't help as the bottleneck is CPU hungry model scoring.

---

### Known issues:

- If a worker dies, but it was processing something, then the caller will infinitely wait for the result!

- Worker monitoring threads run too rarely, it is possible to create a new job after workers have already failed (the window is ~0.2 sec)

- DO NOT terminate workers, might corrupt the queue and shared dict (dies with locks acquired), 
try stopping gracefully, doesnt join within timeout, terminate

---

### TODO:

- Test with your WS project

- Test with proper CPU hungry model

---

### Brainstorming (maybe TODO / extras):

- Check the size of user provided args, kwargs. If they are too large, instead of copying them, put them in a memory store (Apache Arrow, Manager.dict?)
and pass the object ID through the queue? The worker then needs to check if it gets the object or ID of the object.
Consider the MPIRE's approach to copy-on-write (https://github.com/Slimmer-AI/mpire) + excellent read by the author (https://towardsdatascience.com/mpire-for-python-multiprocessing-is-really-easy-d2ae7999a3e9)

- What if user provided callable relies on other objects/clients such as BigQuery client?
