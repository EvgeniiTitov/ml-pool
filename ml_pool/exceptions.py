__all__ = ["UserProvidedCallableFailedError"]


class UserProvidedCallableFailedError(Exception):
    """This exception is raised when either LoadModelCallable or
    ScoreModelCallable function fails"""
