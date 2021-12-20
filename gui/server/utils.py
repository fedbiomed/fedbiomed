import os
import configparser
from flask import jsonify, request
from functools import wraps
from schemas import Validator


def get_node_id(config_file: str):
    """ This method parse given config file and returns node_id
        specified in the node config file.

    Args: 

        config_file     (str): Path for config file of the node that 
                        GUI services will running for
    """

    cfg = configparser.ConfigParser()
    if os.path.isfile(config_file):
        cfg.read(config_file)
    else:
        raise Exception(
            f'Config file does not exist, can not start flask app. Please check follwing folder in your file system {config_file}')

    # Get node id from config file 
    node_id = cfg.get('default', 'node_id')

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
        with a message, it is used for API enpoints  
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



