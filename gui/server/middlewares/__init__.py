from functools import wraps
from typing import Callable


def middleware(middlewares: list[Callable]):
    """Middleware decorator for routes """
    def _middleware(func):
        @wraps(func)
        def __middleware(*args, **kwargs):
            for _function in middlewares:
                result = _function()
                if result:
                    return result
            return func(*args, **kwargs)
        return __middleware
    return _middleware
