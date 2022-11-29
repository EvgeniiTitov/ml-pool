### TODO:

- Test if a worker just fails (raise manually) - hangs the receiving thread
- Pool needs to be manually closed (can automate with atexit?)
- Test with proper model (YOLO or something) - fix loading Torch model
- More thorough testing from multiple threaded callers + API test


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

### Issues / Gotchas:

- IPC in Python is done via pickling / unpickling objects - transferring large objects such as images
could be expensive
- If model scoring latency is measured in 1-10 ms, unloading model scoring / feature engineering to
the pool may only hurt the overall performance, might make more sense to score the model directly
- ! It is common to spin up multiple instances of the API app inside a container using tools such as
gunicorn etc. Be careful when using MLPool in such configuration (might run out of CPU), it's primary task is to unload 
heave model scoring / feature engineering bits from the main process which is the bottleneck to the API
performance. In our case each container runs only a single instance of the API app, spinning up more instances
won't help as the bottleneck is model scoring.

---


### How to use / Examples

User is to provide two callables:

1. A callable that loads a model and returns it (ran by each worker in the pool to load model once for inference):

```python
def load_model(model_path: str):
    model = xgboost.Booster()
    model.load_model(model_path)
    return model
```

2. A callable that scores the model, that must follow the signature (model, *args, **kwargs):

```python
def score_model(model, features):
    time.sleep(0.15)  # Imitates a heavy model that takes time to score
    features = xgboost.DMatrix([features])
    return np.argmax(model.predict(features))
```

Then, the pool could be initialised and used as follows:

```python
from ml_pool import MLPool


def main():
    job_id = pool.schedule_model_scoring(features=[6.2, 2.2, 4.5, 1.5])
    result = pool.get_scoring_result(job_id, wait_if_not_available=True)


if __name__ == '__main__':
    pool = MLPool(
        load_model_func=partial(load_model, "iris_xgb.json"),
        score_model_func=score_model,
        nb_workers=10
    )
    main()
    pool.shutdown()
```
Under the hood, MLPool calls the score_model_func with the model object it gets from the 
load_model_func AND whatever gets passed to .schedule_model_scoring() method. Said that, 
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