import dataclasses
import uuid
import datetime

from ml_pool.custom_types import (
    OptionalArgs,
    OptionalKwargs,
    ScoreModelCallable,
)


__all__ = ["BaseMessage", "JobMessage"]


class BaseMessage:
    pass


@dataclasses.dataclass
class JobMessage(BaseMessage):
    """Represents a message the pool puts in the queue to get processed by
    one of the workers"""

    created_at: datetime.datetime
    message_id: uuid.UUID
    user_func: ScoreModelCallable
    model_name: str
    args: OptionalArgs = None
    kwargs: OptionalKwargs = None
