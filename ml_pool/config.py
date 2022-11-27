class Config:
    LOGGER_VERBOSE = True
    LOGGER_FORMAT = (
        "%(name)s %(process)d %(levelname)s %(lineno)s: %(message)s"
    )

    WORKERS_COUNT = 5
    MESSAGE_QUEUE_SIZE = 100

    USER_CODE_FAILED_EXIT_CODE = 228
    MONITOR_THREAD_SLEEP_TIME = 1.0
