from functools import wraps
from typing import Callable


def middleware(middlewares: list[Callable]):
    """Middleware decorator for routes

    Args:
         middlewares: List of middlewares.
            Middlewares are python functions that does not take any argument, and they have to use
            `g` state object of Flask to pass data to next middlewares or the last route.

            A middleware;

                1 - Does validity check and returns `error` with 400 if request is not valid (bad)
                2 - Reads data/object and update `g` state of the Flask and make it available for the
                    next middlewares or end of the route.
                3 - Should not return anything as long as there is no error during execution or validation.
                    If the purpose of the middleware is to update global state `g` state of the Flask should
                    be updated wıthout returning anything.
    """
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
