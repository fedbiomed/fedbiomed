from functools import wraps
import uuid
import re
import jwt

from hashlib import sha512
from flask import make_response, request
from flask_jwt_extended import (
    create_access_token, verify_jwt_in_request, 
    create_refresh_token, get_jwt_identity,
    set_access_cookies, set_refresh_cookies, 
    unset_jwt_cookies
)
from datetime import datetime, timedelta

from fedbiomed.common.constants import UserRoleType
from gui.server.utils import success, validate_request_data
from gui.server.schemas import ValidateUserFormRequest
from . import api
from utils import error, response
from db import gui_database
from app import app

table = gui_database.db().table('_default')
query = gui_database.query()


def set_password_hash(password: str) -> str:
    """ Method for setting password hash 
    Args: 

            password    (str): Password of the user
    """
    return sha512(password.encode('utf-8')).hexdigest()


def check_password_hash(password: str, user_password_hash: str) -> bool:
    """ Method used to compare password hashes. 
        Used to verify the user password
    Args: 

        password (str): Password to compare against the user password hash
        user_password_hash (str): User password hash
    Returns:
        True if the password hash matches the user password one
        False otherwise
    """
    password_hash = sha512(password.encode('utf-8'))
    return password_hash.hexdigest() == user_password_hash


def get_user_by_mail(user_email: str):
    """ Method used to retrieve a user from the database based on its email
    Args: 

        user_mail (str): The mail of the user to retrieve from the database
    """
    return table.search(query.user_email == user_email)


def check_mail_format(user_mail: str) -> bool:
    """ Method used to check the format of the user email
    Args: 

        user_mail (str): The mail to check
    """
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, user_mail)


def check_password_format(user_password: str) -> bool:
    """ Method used to check the format of the user password
    Args: 

        user_password (str): The password to check. It should be 
        - at least 8 character long
        - with at least one uppercase letter, one lowercase letter and one number
    """
    regex = r'^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9]).{8,}$'
    return re.fullmatch(regex, user_password)


def token_required(f):
    """ Decorator for verifying the JWT token
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        # return 401 if token is not passed
        if not token:
            return error('Unhautorized, please log in'), 401

        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = table.get(query.user_id == data['sub'])
        except Exception as e:
            return error(str(e)), 401
        return f(current_user, *args, **kwargs)

    return decorated



@api.route('/register', methods=['POST'])
@validate_request_data(schema=ValidateUserFormRequest)
def register():
    """ API endpoint to register new user in the database.

    Request {application/json}:
        email (str): Email of the user to register
        password (str): Password of the user to register

    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        409:
            success : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        201:
            success : Boolean value indicates that the request is success
            result  : null
            message : The message for response
    """
    req = request.json
    if not req or not req['email'] or not req['password']:
        return error('Email or password is missing'), 400

    email = req['email']
    password = req['password']

    if not check_mail_format(email):
        return error('Wrong email format'), 400

    if not check_password_format(password):
        return error('Password should be at least 8 character long, with at least one uppercase letter, one lowercase letter and one number'), 400

    if get_user_by_mail(email):
        return error('Email already Present. Please log in'), 409
    try :
        # Create unique id for the user
        user_id = 'user_' + str(uuid.uuid4())
        table.insert({
            "user_email": email,
            "password_hash": set_password_hash(password),
            "user_role": UserRoleType.USER,
            "creation_date": datetime.utcnow().ctime(),
            "user_id": user_id
        })
        res = table.get(query.user_id == user_id)
        return response({
            'user_id': res['user_id'], 
            'user_email': res['user_email']
        }, 'User successfully registered'), 201
    except Exception as e:
        return error(str(e)), 400


@api.route('/token/auth', methods=['POST'])
@validate_request_data(schema=ValidateUserFormRequest)
def login():
    """ API endpoint for logging user in

    Request {application/json}:
        email (str): Email of the user to log in
        password (str): Password of the user to log in

    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        401:
            success : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        200:
            success : Boolean value indicates that the request is success
            result  : null
            message : The message for response
    """
    req = request.json
    if not req or not req['email'] or not req['password']:
        return error('Email or password is missing'), 400
    email = req['email']
    password = req['password']

    user_db = get_user_by_mail(email)
    if not user_db:
        return make_response(
            'Could not verify',
            {'WWW-Authenticate' : 'Basic realm ="User does not exist"'}
        ), 401

    # Should send back only one item
    user = user_db[0]
    if check_password_hash(password, user['password_hash']):
        access_token = create_access_token(identity=user['user_id'], fresh=True)
        refresh_token = create_refresh_token(identity=user['user_id'])
        resp = make_response(success('User successfully logged in'))
        set_access_cookies(resp, access_token)
        set_refresh_cookies(resp, refresh_token)
        return resp, 200

    return make_response(
        'Could not verify',
        {'WWW-Authenticate' : 'Basic realm ="Wrong Password"'}
    ), 401


# TODO: Does not work yet
# @api.route('/token/refresh', methods=['POST'])
# @token_required
# # Should we use refresh any token which is about to expire or let the frontend call the endpoint
# def refresh_expiring_jwts(response):
#     """ API endpoint for refreshing JWT token
#     """
#     verify_jwt_in_request(optional=True)
#     access_token = create_access_token(identity=get_jwt_identity())
#     resp = make_response(success('Access token successfully refreshed'))
#     set_access_cookies(response, access_token)
#     return resp, 200


@api.route('/token/remove', methods=['POST'])
def logout():
    resp = make_response(success('User successfully logged out'))
    unset_jwt_cookies(resp)
    return resp, 200


# TODO : Implement method to retrieve user password
