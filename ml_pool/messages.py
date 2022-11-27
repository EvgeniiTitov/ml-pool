import dataclasses
import uuid

from ml_pool.custom_types import OptionalArgs, OptionalKwargs


__all__ = ["BaseMessage", "JobMessage"]


class BaseMessage:
    pass


@dataclasses.dataclass
class JobMessage(BaseMessage):
    message_id: uuid.UUID
    args: OptionalArgs = None
    kwargs: OptionalKwargs = None
