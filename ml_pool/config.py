import multiprocessing


class Config:
    LOGGER_VERBOSE = True
    LOGGER_FORMAT = (
        "%(name)s %(process)d %(levelname)s %(lineno)s: %(message)s"
    )
    # Assuming one core is used for the API to serve requests
    WORKERS_COUNT = multiprocessing.cpu_count() - 1

    MESSAGE_QUEUE_SIZE = 100
    DEFAULT_MIN_QUEUE_SIZE = 50

    USER_CODE_FAILED_EXIT_CODE = 222
    MONITOR_THREAD_SLEEP_TIME = 1.0

    CANCELLED_JOBS_KEY_NAME = "cancelled_jobs"
    TASK_TTL = 10.0
