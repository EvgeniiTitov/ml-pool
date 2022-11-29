from typing import Any, Callable, Sequence, Dict, Optional
import uuid


__all__ = [
    "LoadModelCallable",
    "ScoreModelCallable",
    "OptionalArgs",
    "OptionalKwargs",
    "ResultDict",
    "MLModel",
]


MLModel = object
LoadModelCallable = Callable[[Optional[Any]], MLModel]
ScoreModelCallable = Callable[[MLModel, Any], Any]
OptionalArgs = Optional[Sequence[Any]]
OptionalKwargs = Optional[Dict[str, Any]]
ResultDict = Dict[uuid.UUID, Any]
