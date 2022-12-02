`Disclamer: THIS IS A NICHE USE CASE AND IS NOT A PROPER WAY TO SOLVE SUCH A PROBLEM`

---

### Use case / why


Cloud Run as of November 2022 does not support GPU machines. Assuming we have a model, 
which is CPU hungry and takes time to score, which we want to serve through an API while:

- keeping overall application complexity low (no dedicated service to serve the model)
- minimising latency
- maximasing throughput (the number of requests a single Cloud Run instance can handle)

How do we do it?

A typical simple API serving a ML model instantiates the model right in the main process, which
results in the model consuming its memory and CPU cycles for scoring, which directly affects the API
performance. It would be neat to unload the model scoring / feature engineering bit to dedicated cores.
This would let us:
- use CPU resources more efficiently (loading more cores instead of a single one)
- avoid hurting API performance by moving model scoring / feature engineering bits that require memory and CPU away from the API process
- decrease latency
- increase throughput 


On top of that, the solution seems to work well with the Cloud Run autoscaling strategy. When 
the load is low, the workers serving the models are pretty much idle. As the load increases,
the workers get busy, which in turn increases overall Cloud Run instance CPU usage. Once
it reaches a certain threshold, Cloud Run will spin up another instance to handle the load.

---

### Gotchas:

- IPC in Python is done via pickling / unpickling objects - transferring large objects such as images
could be expensive

- If your model is light and takes very little time to score, then adding the MLPool will
only slow things down. In this case it makes sense to score the model directly in the API process. 
If there is feature engineering work associated with pulling/preprocessing features, then
it might make sense, depends. 

- ! It is common to spin up multiple instances of the API app inside a container using tools such as
gunicorn etc. Be careful when using MLPool in such configuration as you could overload CPU constantly triggering
Cloud Run to scale up lol. In our case each container runs only a single instance of the API app, spinning up more instances
within the same container won't help as the bottleneck is CPU hungry model scoring.

---


### How to use / Examples

User is to provide two callables:

1. A callable that loads a model and returns it (ran _once_ by each worker in the pool to load model for inference):

```python
def load_model(model_path: str):
    model = xgboost.Booster()
    model.load_model(model_path)
    return model
```

2. A callable that scores the model, that must follow the signature `(model, *args, **kwargs)`:

```python
def score_model(model, features):
    features = xgboost.DMatrix([features])
    return np.argmax(model.predict(features))
```

Then, the pool could be initialised and used as follows:

```python
...
from ml_pool import MLPool

app = FastAPI()


class Request(pydantic.BaseModel):
    features: list[float]


class Response(pydantic.BaseModel):
    prediction: int


@app.post("/predict")
def score(request: Request) -> Response:
    # UNLOAD DATA CRUNCHING CPU HEAVY MODEL SCORING TO THE POOL WITHOUT
    # OVERLOADING THE API PROCESS
    job_id = pool.schedule_scoring(kwargs={"features": request.features})
    result = pool.get_result(job_id, wait_if_unavailable=True)
    return Response(prediction=result)


if __name__ == '__main__':
    with MLPool(
        load_model_func=partial(load_model, "xgb.json"),
        score_model_func=score_model,
        nb_workers=4
    ) as pool:
        uvicorn.run(app)
```

Under the hood, MLPool calls the provided _score_model_func_ with the model object it gets from the 
_load_model_func_ AND whatever gets passed to .schedule_scoring() method. As a result, 
the user has full control of what they want to run on the pool.


--- 

### Benchmarks

- APIs Fake Load: sync (examples/sync_api.py) VS pool based (examples/ml_pool_api.py)

1. 1 uvicorn worker, 10 concurrent clients, 50 requests / client, 10M CPU burn cycles (imitates model scoring)

```
sync - 338 seconds
ml_pool - 84 seconds (11 workers)
```

2. 1 uvicorn worker, 20 concurrent clients, 50 requests / client, 10M CPU burn cycles
```
sync - 657 seconds (1.5 requests / s)
ml_pool - 143 seconds (11 workers) (7 requests/s)
```


- YOLO (TODO)


---


### TODO:

- Result dict needs to be cleaned if the caller never consumes the result (TTL for the result?)

- Ability to provide multiple objects (models to load). Like a KV with model name and a callable to load it.

- When scheduling model scoring, provide the function for scoring (instead of passing it in the constructor), args and
the loaded model to use (as the first parameter)

- Release as a package

- Test with proper model (YOLO or something) - fix loading Torch model

- Test the pool with async code (use the flags block_until_scheduled AND wait_if_unavailable)

- Feature: Redesign workers monitoring and raising the exception if they failed

  - Test if a worker just fails (raise manually) - hangs the monitor thread

  - Monitoring thread runs too rarely, workers fail, but new jobs get accepted as the flag doesnt get switched cuz the thread is sleeping...
      - Create a function that checks if workers healthy? Could be reused by the monitor + before
    adding new jobs.

- Test with your WS project


### Brainstorming (maybe TODO):

- Check the size of user provided args, kwargs. If they are too large, instead of copying them, put them in a memory store (Apache Arrow, Manager.dict?)
and pass the object ID through the queue? The worker then needs to check if it gets the object or ID of the object.
Consider the MPIRE's approach to copy-on-write (https://github.com/Slimmer-AI/mpire) + excellent read by the author (https://towardsdatascience.com/mpire-for-python-multiprocessing-is-really-easy-d2ae7999a3e9)