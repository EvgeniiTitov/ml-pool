import dataclasses
import uuid

from ml_pool.custom_types import OptionalArgs, OptionalKwargs


__all__ = ["BaseMessage", "JobMessage"]


class BaseMessage:
    pass


@dataclasses.dataclass
class JobMessage(BaseMessage):
    """Represents a message the pool puts in the queue to get processed by
    one of the workers"""

    message_id: uuid.UUID
    args: OptionalArgs = None
    kwargs: OptionalKwargs = None
