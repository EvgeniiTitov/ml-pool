from typing import Any, Callable, Sequence, Dict, Optional, Union
import uuid


__all__ = [
    "LoadModelCallable",
    "ScoreModelCallable",
    "OptionalArgs",
    "OptionalKwargs",
    "SharedDict",
    "MLModel",
]


MLModel = Any
LoadModelCallable = Callable[[Optional[Any]], MLModel]
ScoreModelCallable = Callable[[MLModel, Any], Any]
OptionalArgs = Optional[Sequence[Any]]
OptionalKwargs = Optional[Dict[str, Any]]
SharedDict = Dict[Union[str, uuid.UUID], Any]
