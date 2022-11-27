class Config:
    LOGGER_VERBOSE = True
    LOGGER_FORMAT = "%(name)s %(lineno)s: %(message)s"

    WORKERS_COUNT = 3
    MESSAGE_QUEUE_SIZE = 10

    USER_CODE_FAILED_EXIT_CODE = 228
    MONITOR_THREAD_SLEEP_TIME = 1.0
