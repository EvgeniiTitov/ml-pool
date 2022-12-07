### Use case / why


Cloud Run as of November 2022 does not support GPU machines. Assuming we have a model, 
which is CPU hungry and takes time to score, which we want to serve through an API/web sockets while:

- keeping overall application complexity low (no dedicated service to serve ML models)
- minimising latency
- maximasing throughput (the number of requests a single Cloud Run instance can handle)


A typical simple API serving ML model(s) instantiates a model right in the main process, which
results in the model consuming the process's memory and CPU cycles for scoring. This directly 
affects the API performance. Ideally, we want to move the model scoring / feature engineering bit 
to dedicated cores. This would let us:

- use CPU resources more efficiently (loading more cores instead of a single one provided that the container has 1+ cores at its disposal)
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


TBA 


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
