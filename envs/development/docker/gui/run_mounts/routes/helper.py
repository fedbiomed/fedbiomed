from flask import json, jsonify, request
from werkzeug.exceptions import BadRequest
from functools import wraps
from .schemas import Validator

def error(msg:str):

    """ Function that returns jsonfied error result
        it is used for API enpoints  
    Args:

        msg     (str): Response message for failed request.
    """
    return jsonify(
        {
            'success' : False,
            'message' : msg
        }
    )

def success(msg:str):


    """ Function that returns jsonfied success result
        with a message, it is used for API enpoints  
    Args:

        msg     (str): Response message for successful request.
    """

    return jsonify(
        {
            'success' : True,
            'message' : msg
        }
    )


def validate_json(function):

    """ Validate requested JSON whether is in 
        correct format"""

    @wraps(function)
    def wrapper(*args, **kw):

        if request.headers.get('Content-Type') != 'application/json':
            res = error("Reques body should be application/json")
            return res, 400
        elif request.json is None:
            res = error("application/json returns `None`")
            return res, 400
  
        # Otherwise keep executing route controller    
        return function(*args, **kw)

    return wrapper

def validate_request_data(schema: Validator):

    """ Validate reqeusted data. This wrapper method gets schema
        and applies validation based on provided information 
        in schema

        Args: 
            schema (Validator) : Schema class to to check inputs in 
                                request object
    """
    def decorator(controller):
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