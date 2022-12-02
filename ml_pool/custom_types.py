from typing import Any, Callable, Sequence, Dict, Optional, Union
import uuid


__all__ = [
    "LoadModelCallable",
    "ScoreModelCallable",
    "OptionalArgs",
    "OptionalKwargs",
    "SharedDict",
    "MLModel",
    "MLModels",
    "LoadedMLModels",
]


MLModel = Any  # TODO Consider using an interface
LoadModelCallable = Callable[[Optional[Any]], MLModel]
MLModels = Dict[str, LoadModelCallable]
LoadedMLModels = Dict[str, MLModel]
OptionalArgs = Optional[Sequence[Any]]
OptionalKwargs = Optional[Dict[str, Any]]
ScoreModelCallable = Callable[[MLModel, OptionalArgs, OptionalKwargs], Any]
SharedDict = Dict[Union[str, uuid.UUID], Any]
