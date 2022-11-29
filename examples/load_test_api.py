import sys

sys.path.append("..")

import requests
import threading

from ml_pool.utils import timer


URL = "http://127.0.0.1:8000/iris"
CLIENTS = 20
REQUESTS_PER_CLIENT = 50


def client(index, features):
    for i in range(REQUESTS_PER_CLIENT):
        response = requests.post(url=URL, json={"features": features})
        print(
            f"Client {index} got {i} / {REQUESTS_PER_CLIENT} "
            f"response {response.json()}"
        )


@timer
def main():
    threads = [
        threading.Thread(target=client, args=(i, [6.2, 2.2, 4.5, 1.5]))
        for i in range(CLIENTS)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()