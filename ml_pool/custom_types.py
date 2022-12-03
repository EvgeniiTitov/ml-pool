from typing import Any, Callable, Sequence, Dict, Optional
import uuid


__all__ = [
    "LoadModelCallable",
    "ScoreModelCallable",
    "OptionalArgs",
    "OptionalKwargs",
    "ResultDict",
    "MLModel",
    "MLModels",
    "LoadedMLModels",
    "CancelledDict",
]


MLModel = Any  # TODO Consider using an interface
LoadModelCallable = Callable[[Optional[Any]], MLModel]
MLModels = Dict[str, LoadModelCallable]
LoadedMLModels = Dict[str, MLModel]
OptionalArgs = Optional[Sequence[Any]]
OptionalKwargs = Optional[Dict[str, Any]]
ScoreModelCallable = Callable[[MLModel, OptionalArgs, OptionalKwargs], Any]
ResultDict = Dict[uuid.UUID, Any]
CancelledDict = Dict[uuid.UUID, Any]
