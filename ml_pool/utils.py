import functools
import typing as t
import time
import uuid


__all__ = ["timer", "get_new_job_id"]


def timer(func: t.Callable) -> t.Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> t.Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(
            f"Function {func.__name__} took "
            f"{time.perf_counter() - start: .3f} seconds"
        )
        return result

    return wrapper


def get_new_job_id() -> uuid.UUID:
    return uuid.uuid4()
