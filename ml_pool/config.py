import multiprocessing


class Config:
    LOGGER_VERBOSE = True
    LOGGER_FORMAT = (
        "%(asctime)s %(name)s %(process)d %(levelname)s "
        "%(lineno)s: %(message)s"
    )

    # Assuming one core is used for the API to serve requests
    WORKERS_COUNT = multiprocessing.cpu_count() - 1

    # Jobs / results management
    MESSAGE_QUEUE_SIZE = 100
    DEFAULT_MIN_QUEUE_SIZE = 50
    RESULT_TTL = 300.0  # Seconds

    # Background threads
    MONITOR_THREAD_FREQUENCY = 0.1
    CLEANER_THREAD_FREQUENCY = 10.0

    # Worker errors
    LOAD_MODEL_CALLABLE_FAILED = 222
    LOAD_MODEL_CALLABLE_RETURNED_NOTHING = 223
    SCORE_MODEL_CALLABLE_FAILED = 224
    UNKNOWN_ML_MODEL_REQUESTED = 225
    CUSTOM_EXIT_CODES_MAPPING = {
        LOAD_MODEL_CALLABLE_FAILED: "Failed to load model",
        LOAD_MODEL_CALLABLE_RETURNED_NOTHING: (
            "Load model callable returned invalid object"
        ),
        SCORE_MODEL_CALLABLE_FAILED: (
            "Failed while scoring using provided callable"
        ),
        UNKNOWN_ML_MODEL_REQUESTED: (
            "Unknown (not loaded) model requested for scoring"
        ),
    }
