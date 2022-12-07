import sys
import random
import time

sys.path.append("..")

import requests
import threading

from ml_pool.utils import timer


PATH = "iris"
URL = f"http://127.0.0.1:8000/{PATH}"
CLIENTS = 20
REQUESTS_PER_CLIENT = 100
FEATURES = [
    [3.0, 2.0, 1.0, 0.2],
    [4.9, 2.2, 3.8, 1.1],
    [5.3, 2.5, 4.6, 1.9],
    [6.2, 2.2, 4.5, 1.5],
]

LOCK = threading.Lock()  # Better safe than sorry
WORKER_LATENCIES = {}


def client(index) -> None:
    times = []
    for i in range(REQUESTS_PER_CLIENT):
        start = time.perf_counter()
        response = requests.post(
            url=URL, json={"features": random.choice(FEATURES)}, timeout=20.0
        )
        times.append(time.perf_counter() - start)
        print(
            f"Client {index} got {i} / {REQUESTS_PER_CLIENT} "
            f"response {response.json()}"
        )
    with LOCK:
        WORKER_LATENCIES[index] = times


def calculate_latency() -> float:
    all_latencies = 0
    for value in WORKER_LATENCIES.values():
        all_latencies += sum(value)
    return all_latencies / (CLIENTS * REQUESTS_PER_CLIENT)


@timer
def main():
    threads = [
        threading.Thread(target=client, args=(i,)) for i in range(CLIENTS)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Average response latency: {calculate_latency(): .4f}")


if __name__ == "__main__":
    main()
