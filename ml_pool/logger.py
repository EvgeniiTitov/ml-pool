import functools
import logging
import typing as t
import sys

from ml_pool.config import Config


__all__ = ["get_logger"]


@functools.lru_cache()
def get_logger(name: t.Optional[str] = None) -> logging.Logger:
    def _get_console_handler() -> logging.StreamHandler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        return console_handler

    if not name:
        name = __name__

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(
        logging.DEBUG if Config.LOGGER_VERBOSE else logging.WARNING
    )
    formatter = logging.Formatter(Config.LOGGER_FORMAT)
    logger.addHandler(_get_console_handler())
    return logger
