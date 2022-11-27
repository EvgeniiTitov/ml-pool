import functools
import typing as t
import time


__all__ = ["timer"]


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
