from functools import wraps

from cachelib import FileSystemCache
from flask import request


class RepositoryCache(object):
    file_sizes = {}

    @classmethod
    def clear(cls, path):
        if path in cls.file_sizes:
            del cls.file_sizes[path]

    @classmethod
    def clear_all(cls):
        cls.file_sizes = {}


CACHE_TIMEOUT = 300
cache = FileSystemCache("./__pycache__")


def cached(key: str, prefix: str = "", timeout: int = CACHE_TIMEOUT):
    def _decorator(func):
        @wraps(func)
        def __decorator(*args, **kwargs):
            cache_id = prefix + "-" + request.json[key]
            response = cache.get(cache_id)
            if response is None:
                response = func(*args, **kwargs)
                cache.set(cache_id, response, timeout)
            return response

        return __decorator

    return _decorator
