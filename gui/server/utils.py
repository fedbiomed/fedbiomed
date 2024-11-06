import configparser
import datetime
import os
from functools import wraps
from hashlib import sha512
from flask import jsonify, request

from .cache import RepositoryCache
from .schemas import Validator


def set_password_hash(password: str) -> str:
    """ Method for setting password hash
    Args:

        password (str): Password of the user
    """
    return sha512(password.encode('utf-8')).hexdigest()


def get_node_id(config_file: str):
    """ This method parse given config file and returns node_id
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
            f'Config file does not exist, can not start flask app. Please check following path exists in your file '
            f'system {config_file}')

    # Get node id from config file
    node_id = cfg.get('default', 'id')

    return node_id


def error(msg: str):
    """ Function that returns jsonfied error result
        it is used for API enpoints
    Args:

        msg     (str): Response message for failed request.
    """
    return jsonify(
        {
            'success': False,
            'result': None,
            'message': msg
        }
    )


def success(msg: str):
    """ Function that returns jsonfied success result
        with a message, it is used for API endpoints
    Args:

        msg     (str): Response message for successful request.
    """

    return jsonify(
        {
            'success': True,
            'result': None,
            'message': msg
        }
    )


def response(data: dict, message: str = None):
    """ Global response function that returns jsonfied
        dictionary. It is used when the API endpoint returns
        data.

    Args:
        data (dict): Data that will be sent as a response of the
                      API
        endpoint (str): API endpoint
        message (str):

    """

    res = {
        'success': True,
        'result': data,
        'message': message
    }

    return jsonify(res)


def validate_json(function):
    """ Decorator for validating requested JSON whether is in
        correct JSON format
    Args:
          function (func) : Controller (router)
    """

    @wraps(function)
    def wrapper(*args, **kw):

        if request.headers.get('Content-Type') != 'application/json':
            res = error("Request body should be application/json")
            return res, 400
        elif request.json is None:
            res = error("application/json returns `None`")
            return res, 400

        # Otherwise, keep executing route controller
        return function(*args, **kw)

    return wrapper


def validate_request_data(schema: Validator):
    """ Validate reqeusted data. This wrapper method gets schema
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
    """ Returns creation date and size information of
        given path.

        Args:
            path (str): Absolute path to folder or file
            refresh (bool): If it is true clear cached size value for given path
    """
    stats = os.stat(path)
    creation_time = datetime.datetime.fromtimestamp(stats.st_ctime).strftime('%d/%m/%Y %H:%M')
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

    return creation_time, disk_usage


def get_disk_usage(path: str):
    """ Calculates disk usage of given path
    Args:

        path (str) : Absolute path of file or folder
    """

    size = 0

    if os.path.isfile(path):
        try:
            size = os.path.getsize(path)
        except:
            pass
    elif os.path.isdir(path):
        try:
            for index, (path, dirs, files) in enumerate(os.walk(path)):
                for i, f in enumerate(files):
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)
        except:
            pass

    return parse_size(size)


def parse_size(size):
    """ This function will convert bytes into a human readable form

    Args:
        size (float): File size in KB
    """

    formatter = "%.1f %s"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YT']:
        if size < 1024.0:
            return formatter % (size, unit)
        size /= 1024.0

    return formatter % (size, 'BB')


