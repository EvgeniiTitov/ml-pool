import sys
import random

sys.path.append("..")

import requests
import threading

from ml_pool.utils import timer


PATH = "iris"
URL = f"http://127.0.0.1:8000/{PATH}"
CLIENTS = 25
REQUESTS_PER_CLIENT = 1000

FEATURES = [
    [3.0, 2.0, 1.0, 0.2],
    [4.9, 2.2, 3.8, 1.1],
    [5.3, 2.5, 4.6, 1.9],
    [6.2, 2.2, 4.5, 1.5],
]


def client(index) -> None:
    for i in range(REQUESTS_PER_CLIENT):
        response = requests.post(
            url=URL, json={"features": random.choice(FEATURES)}, timeout=20.0
        )
        print(
            f"Client {index} got {i} / {REQUESTS_PER_CLIENT} "
            f"response {response.json()}"
        )


@timer
def main():
    threads = [
        threading.Thread(target=client, args=(i,)) for i in range(CLIENTS)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
