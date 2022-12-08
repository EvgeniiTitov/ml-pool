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

In order to use MLPool, for every ML model a user wants to run on the pool, they need to provide 
a callable, which loads a model into memory and prepares it. Under the hood, MLPool will ensure 
every worker in the pool running in a dedicated process loads the model(s) in memory.

Consider these functions that load the models.

```python
def load_iris(model_path: str) -> HungryIris:
    # TODO: Loading and preparing ML model goes here 
    return HungryIris(model_path)


def load_diabetes_classifier(model_path: str) -> HungryDiabetesClassifier:
    # TODO: Loading and preparing ML model goes here
    return HungryDiabetesClassifier(model_path)
```

When instantiating MLPool, a user needs to provide a dictionary (models_to_load) where the keys are
model names MLPool will serve, and the values are the callables loading the models (functions above).
In addition, the user can specify the number of workers they want, scoring results TTL etc.

```python
if __name__ == "__main__":
    with MLPool(
        models_to_load={
            "iris": partial(load_iris, "../models/iris_xgb.json"),
            "diabetes_classifier": partial(
                load_diabetes_classifier, "../models/diabetes_xgb.json"
            )
        },
        nb_workers=5,
    ) as pool:
        ...
```



# STOPPED HERE - SCORING


When it comes to scoring, MLPool provides the ability to:

- Create a scoring job on the pool
- Get results of the scoring job
- Cancel a scoring job






- `create_job` or asyncio friendly `create_job_async` methods to execute model on the pool. The methods
expect a callable which accepts as the first parameter a ML model (passed by MLPool) and *args, **kwargs. 

Consider an example

```python
def score_iris(model: HungryIris, features):
    # TODO: Feature engineering etc goes here

    return model.predict(features)
```

```python
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
```
---

### Gotchas:

- IPC in Python is done via pickling / unpickling objects - transferring large objects such as images
could be expensive.

- If your model is light and takes very little time to score (tens of milliseconds), then introducing MLPool will
only slow things down. In this case it makes sense to score the model directly in the API process. 
If there is feature engineering work associated with pulling/preprocessing features, then
it might make sense, depends. 

- ! It is common to spin up multiple instances of the API app inside a container using tools such as
gunicorn etc. Be careful when using MLPool in such configuration as you could overload CPU constantly triggering
Cloud Run to scale up lol. In our case each container runs only a single instance of the API app, spinning up more instances
within the same container won't help as the bottleneck is CPU hungry model scoring.


---


### Known issues:

- If a worker dies, but it was processing something, then the caller will infinitely wait for the result!

- Worker monitoring threads run too rarely, it is possible to create a new job after workers have already failed (the window is ~0.2 sec)


---


### TODO:


- DO NOT terminate workers, might corrupt the queue and shared dict (dies with locks acquired), try stopping gracefully, doesnt join within timeout, terminate

- Release as a package

- Test with proper model (YOLO or something) - fix loading Torch model

- Test the pool with async code (use the flags block_until_scheduled AND wait_if_unavailable)

- Test with your WS project


---

### Brainstorming (maybe TODO / extras):

- Check the size of user provided args, kwargs. If they are too large, instead of copying them, put them in a memory store (Apache Arrow, Manager.dict?)
and pass the object ID through the queue? The worker then needs to check if it gets the object or ID of the object.
Consider the MPIRE's approach to copy-on-write (https://github.com/Slimmer-AI/mpire) + excellent read by the author (https://towardsdatascience.com/mpire-for-python-multiprocessing-is-really-easy-d2ae7999a3e9)

- What if user provided callable relies on other objects/clients such as BigQuery client?
