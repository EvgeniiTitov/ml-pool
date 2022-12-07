import sys
import threading
import time
import random

sys.path.append("..")

import requests

from ml_pool.utils import timer


URL = "http://127.0.0.1:8000/{path}"

IRIS_CLIENTS_NB = 10
IRIS_FEATURES = [
    [3.0, 2.0, 1.0, 0.2],
    [4.9, 2.2, 3.8, 1.1],
    [5.3, 2.5, 4.6, 1.9],
    [6.2, 2.2, 4.5, 1.5],
]
REQUESTS_PER_IRIS_CLIENT = 50

DIABETES_CLIENTS_NB = 10
DIABETES_FEATURES = [
    [4, 184, 78, 39, 277, 37.0, 0.264, 31],
    [0, 94, 0, 0, 0, 0.0, 0.256, 25],
    [1, 181, 64, 30, 180, 34.1, 0.328, 38],
    [0, 135, 94, 46, 145, 40.6, 0.284, 26],
    [1, 95, 82, 25, 180, 35.0, 0.233, 43],
]
REQUESTS_PER_DIABETES_CLIENT = 50

LOCK = threading.Lock()  # Better safe than sorry
WORKER_LATENCIES = {"iris": {}, "diabetes": {}}


def client_iris(index: int) -> None:
    times = []
    for i in range(REQUESTS_PER_IRIS_CLIENT):
        start = time.perf_counter()
        response = requests.post(
            url=URL.format(path="iris"),
            json={"features": random.choice(IRIS_FEATURES)},
            timeout=20.0,
        )
        times.append(time.perf_counter() - start)
        print(
            f"Iris client {index} got {i} / {REQUESTS_PER_IRIS_CLIENT} "
            f"response {response.json()}"
        )
    with LOCK:
        WORKER_LATENCIES["iris"][index] = times


def client_diabetes(index: int) -> None:
    times = []
    for i in range(REQUESTS_PER_DIABETES_CLIENT):
        start = time.perf_counter()
        response = requests.post(
            url=URL.format(path="diabetes"),
            json={"features": random.choice(DIABETES_FEATURES)},
            timeout=20.0,
        )
        times.append(time.perf_counter() - start)
        print(
            f"Diabetes client {index} got {i} / {REQUESTS_PER_DIABETES_CLIENT}"
            f" response {response.json()}"
        )
    with LOCK:
        WORKER_LATENCIES["diabetes"][index] = times


def calculate_latencies():
    all_latencies = 0
    iris_latencies = 0
    diabetes_latencies = 0

    for key in WORKER_LATENCIES.keys():
        latencies = 0
        for value in WORKER_LATENCIES[key].values():
            latencies += sum(value)

        all_latencies += latencies
        if key == "iris":
            iris_latencies += latencies
        else:
            diabetes_latencies += latencies

    iris_latency = iris_latencies / (
        IRIS_CLIENTS_NB * REQUESTS_PER_IRIS_CLIENT
    )  # noqa
    diabetes_latency = diabetes_latencies / (
        DIABETES_CLIENTS_NB * REQUESTS_PER_DIABETES_CLIENT
    )  # noqa
    total_latency = all_latencies / (
        IRIS_CLIENTS_NB * REQUESTS_PER_IRIS_CLIENT
        + DIABETES_CLIENTS_NB * REQUESTS_PER_DIABETES_CLIENT
    )  # noqa
    return total_latency, iris_latency, diabetes_latency


@timer
def main():
    iris_client_threads = [
        threading.Thread(target=client_iris, args=(i,))
        for i in range(IRIS_CLIENTS_NB)
    ]
    diabetes_client_threads = [
        threading.Thread(target=client_diabetes, args=(i,))
        for i in range(DIABETES_CLIENTS_NB)
    ]
    clients = [*iris_client_threads, *diabetes_client_threads]

    for client in clients:
        client.start()

    for client in clients:
        client.join()

    total_latency, iris_latency, diabetes_latency = calculate_latencies()
    print("Total latency:", total_latency)
    print("Iris latency:", iris_latency)
    print("Diabetes latency:", diabetes_latency)


if __name__ == "__main__":
    main()
