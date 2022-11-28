#### TODO:

- Test if user provided callables fail - what happen then? How to communicate exception from the monitor thread to the main process? Close the pool?
- Test if a worker just fails (raise manually) - hangs the receiving thread
- Test with proper model (YOLO or something)
- More thorough testing from multiple threaded callers + API test


---

#### Use case / why

Cloud Run as of November 2022 does not support GPU machines. Assuming we have a model, 
which is CPU hungry that we want to serve through an API while keeping overall
application complexity low (no dedicated service to serve the model), minimising latency and
maximasing throughput, how do we do it?

Typical simple API serving a ML model instantiates the model right in the main process, which
consumes its memory and CPU cycles for scoring, which directly affects the API
performance. It would be neat to unload model scoring to dedicated cores which would
allow use CPU resources more efficiently, avoid hurting API performance by loading the 
API process with model scoring, and increase API throughput while minimising latency, as there
could be multiple instances of the model running on in the pool. 

On top of that, the solution seems to work well with the Cloud Run autoscaling strategy. When
workers serving models are idle, the overall cloud run instance CPU usage is low, its just
the API process. As the load increases, at some point the instance CPU usage will reach
the point, which will trigger another instance to be created to handle the load.

---


#### How to use

---

#### Example

--- 

#### Prod results

