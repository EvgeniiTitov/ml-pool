__all__ = [
    "UserProvidedCallableFailedError",
    "JobWithSuchIDDoesntExistError",
    "UserProvidedCallableError",
]


class MLPoolError(Exception):
    pass


class UserProvidedCallableFailedError(MLPoolError):
    """This exception is raised when either LoadModelCallable or
    ScoreModelCallable function fails"""


class JobWithSuchIDDoesntExistError(MLPoolError):
    """This exception is raised when a user attempts to get scoring results
    using unknown id (such job was never created in the first place)"""


class UserProvidedCallableError(MLPoolError):
    """This exception is raised if something is wrong with the user provided
    callable"""
