from typing import Any, Callable, Sequence, Dict
import abc


__all__ = [
    "LoadModelCallable",
    "ScoreModelCallable",
    "BaseMLModel",
    "OptionalArgs",
    "OptionalKwargs"
]


class BaseMLModel(abc.ABC):
    """Represents any ML model that will be loaded and run by the pool"""

    @abc.abstractmethod
    def score(self, *args, **kwargs) -> Any:
        ...


LoadModelCallable = Callable[[], BaseMLModel]
ScoreModelCallable = Callable[[BaseMLModel, Any], Any]
OptionalArgs = Sequence[Any]
OptionalKwargs = Dict[str, Any]
