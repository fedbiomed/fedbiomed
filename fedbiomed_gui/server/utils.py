import configparser
import datetime
import os
import time
from collections import deque
from functools import wraps
from hashlib import sha512
from pathlib import Path
from typing import Union

from flask import jsonify, request

from .cache import RepositoryCache
from .schemas import Validator


def set_password_hash(password: str) -> str:
    """Method for setting password hash
    Args:

        password (str): Password of the user
    """
    return sha512(password.encode("utf-8")).hexdigest()


def get_node_id(config_file: str):
    """This method parse given config file and returns node_id
        specified in the node config file.

    Args:

        config_file     (str): Path for config file of the node that
                        GUI services will run for
    """

    cfg = configparser.ConfigParser()
    if os.path.isfile(config_file):
        cfg.read(config_file)
    else:
        raise Exception(
            f"Config file does not exist, can not start flask app. Please check following path exists in your file "
            f"system {config_file}"
        )

    # Get node id from config file
    node_id = cfg.get("default", "id")

    return node_id


def error(msg: str):
    """Function that returns jsonfied error result
        it is used for API endpoints
    Args:

        msg     (str): Response message for failed request.
    """
    return jsonify({"success": False, "result": None, "message": msg})


def success(msg: str):
    """Function that returns jsonfied success result
        with a message, it is used for API endpoints
    Args:

        msg     (str): Response message for successful request.
    """

    return jsonify({"success": True, "result": None, "message": msg})


def response(data: dict, message: str = None):
    """Global response function that returns jsonfied
        dictionary. It is used when the API endpoint returns
        data.

    Args:
        data (dict): Data that will be sent as a response of the
                      API
        endpoint (str): API endpoint
        message (str):

    """

    res = {"success": True, "result": data, "message": message}

    return jsonify(res)


def validate_json(function):
    """Decorator for validating requested JSON whether is in
        correct JSON format
    Args:
          function (func) : Controller (router)
    """

    @wraps(function)
    def wrapper(*args, **kw):
        if request.headers.get("Content-Type") != "application/json":
            res = error("Request body should be application/json")
            return res, 400
        elif request.json is None:
            res = error("application/json returns `None`")
            return res, 400

        # Otherwise, keep executing route controller
        return function(*args, **kw)

    return wrapper


def validate_request_data(schema: Validator):
    """Validate requested data. This wrapper method gets schema
    and applies validation based on provided information
    in schema

    Args:
        schema (Validator) : Schema class to check inputs in
                            request object
    """

    def decorator(controller):
        """
        Decorator to compare request JSON with
        given json schema

        Args:
             controller (func): Controller function for the route
        """

        @wraps(controller)
        def wrapper(*args, **kw):
            try:
                inputs = schema(request)
                inputs.validate()
            except Exception as e:
                return error(str(e)), 400

            return controller(*args, **kw)

        return wrapper

    return decorator


def file_stats(path: str, refresh: bool = False):
    """Returns creation date and size information of
    given path.

    Args:
        path (str): Absolute path to folder or file
        refresh (bool): If it is true clear cached size value for given path
    """
    try:
        stats = os.stat(path)
        creation_time = datetime.datetime.fromtimestamp(stats.st_ctime).strftime(
            "%d/%m/%Y %H:%M"
        )
    except (PermissionError, FileNotFoundError, OSError):
        # Cannot stat this path — system/locked/unreadable file
        return None, "N/A"
    try:
        if refresh:
            disk_usage = get_disk_usage(path)
            RepositoryCache.clear(path)
            RepositoryCache.file_sizes[path] = disk_usage
        else:
            if path in RepositoryCache.file_sizes:
                disk_usage = RepositoryCache.file_sizes[path]
            else:
                disk_usage = get_disk_usage(path)
                RepositoryCache.file_sizes[path] = disk_usage
    except Exception:
        disk_usage = "N/A"

    return creation_time, disk_usage


def get_disk_usage(
    path: Union[str, Path],
    *,
    max_depth: int = 1,
    max_files: int = 10_000,
    time_budget_s: float = 0.5,
) -> str:
    """
    Fast, bounded disk-usage estimate for file/dir.
    Trades accuracy for speed with depth/file/time caps.
    Returns a human-readable string via parse_size().
    """
    p = Path(path)
    size = 0
    start = time.time()

    def timed_out() -> bool:
        return (time.time() - start) > time_budget_s

    try:
        if p.is_file():
            try:
                size = p.stat().st_size
            except (OSError, PermissionError, FileNotFoundError):
                size = 0
        elif p.is_dir():
            files_seen = 0
            q = deque([(p, 0)])

            while q and not timed_out() and files_seen < max_files:
                cur, depth = q.popleft()
                # Depth cap
                if depth > max_depth:
                    continue

                try:
                    with os.scandir(cur) as it:
                        for entry in it:
                            if timed_out() or files_seen >= max_files:
                                break
                            try:
                                # Skip symlinks/junctions
                                if entry.is_symlink():
                                    continue
                                if entry.is_file(follow_symlinks=False):
                                    try:
                                        size += entry.stat(
                                            follow_symlinks=False
                                        ).st_size
                                        files_seen += 1
                                    except (
                                        OSError,
                                        PermissionError,
                                        FileNotFoundError,
                                    ):
                                        pass
                                elif (
                                    entry.is_dir(follow_symlinks=False)
                                    and depth < max_depth
                                ):
                                    q.append((Path(entry.path), depth + 1))
                            except (OSError, PermissionError, FileNotFoundError):
                                continue
                except (OSError, PermissionError, FileNotFoundError):
                    # unreadable dir; skip
                    continue
        else:
            size = 0
    except Exception:
        size = 0

    return parse_size(size)


def parse_size(size):
    """This function will convert bytes into a human readable form

    Args:
        size (float): File size in KB
    """

    formatter = "%.1f %s"
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YT"]:
        if size < 1024.0:
            return formatter % (size, unit)
        size /= 1024.0

    return formatter % (size, "BB")
