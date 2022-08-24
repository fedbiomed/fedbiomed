import re
import uuid
from datetime import datetime
from functools import wraps
from hashlib import sha512

from db import user_database
from flask import request
from flask_jwt_extended import (jwt_required, create_access_token, create_refresh_token, unset_jwt_cookies,
                                verify_jwt_in_request, get_jwt)
from utils import error, response
from fedbiomed.common.constants import UserRoleType, UserRequestStatus
from gui.server.schemas import ValidateUserFormRequest
from gui.server.utils import validate_request_data
from . import api

user_table = user_database.table('Users')
user_requests_table = user_database.table('Requests')
query = user_database.query()


def set_password_hash(password: str) -> str:
    """ Method for setting password hash 
    Args: 

        password (str): Password of the user
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


def get_user_by_email(user_email: str) -> str:
    """ Method used to retrieve a user from the database based on its email
    Args: 

        user_email (str): The mail of the user to retrieve from the database
    """
    return user_table.search(query.user_email == user_email)


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


def admin_required(func):
    """Decorator used to protect endpoints that require admin role"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()
        if claims['role'] != UserRoleType.ADMIN:
            return error("You don't have permission to perform this action ! Please contact your "
                         "local Administrator"), 403
        else:
            return func(*args, **kwargs)

    return wrapper


@api.route('/update-password', methods=['POST'])
@jwt_required()
@validate_request_data(schema=ValidateUserFormRequest)
def update_password():
    """ API endpoint to update user's password.
    Before changing password, checks if User email in JSON is the same stored in the JWT 
    
    Request {application/json}:
        email (str): user's email
        password (str): new password user wants to update
    
    Response {application/json}:
        400:
            error   : Boolean error status (False)
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
    email, password = req['email'], req['password']

    if not check_password_format(password):
        return error('Password should be at least 8 character long, with at least one uppercase '
                     'letter, one lowercase letter and one number'), 400
    decoded_json = get_jwt()

    if decoded_json['email'] != email:
        # TODO: allow also operation if user's role is admin
        return error('Error invalid user id'), 400

    user_name = get_user_by_email(email)
    if not user_name:
        return error('Invalid operation: User does not belong to database'), 400

    try:
        user_table.update({
            "user_email": email,
            "password_hash": set_password_hash(password)
        })
        res = user_table.get(query.user_email == email)

        return response({
            'user_id': res['user_id'],
            'user_email': email}, 'User password successfully updated'), 200

    except Exception as e:
        return error(str(e)), 400


@api.route('/register', methods=['POST', 'GET'])
@validate_request_data(schema=ValidateUserFormRequest)
def register():
    """ API endpoint to register new user in the database (as a simple user).

    Request {application/json}:
        email (str): Email of the user to register
        password (str): Password of the user to register
        name (str): Name of the user to register
        surname (str): Surname of the user to register

    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        409:
            success : Boolean error status (False)
            result  : null
            message : Message about error, when user is already registered but wants to register
                        under another account
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
    name = req['name']
    surname = req['surname']

    if not check_mail_format(email):
        return error('Wrong email format'), 400

    if not check_password_format(password):
        return error(
            'Password should be at least 8 character long, with at least one uppercase letter, one lowercase letter and one number'), 400

    if get_user_by_email(email):
        return error('Email already Present. Please log in'), 409
    try:
        # Create unique id for the request
        request_id = 'request_' + str(uuid.uuid4())
        user_requests_table.insert({
            "user_name": name,
            "user_surname": surname,
            "user_email": email,
            "password_hash": set_password_hash(password),
            "user_role": UserRoleType.USER,
            "creation_date": datetime.utcnow().ctime(),
            "request_id": request_id,
            "request_status": UserRequestStatus.NEW
        })
        res = user_requests_table.get(query.request_id == request_id)
        return response({
            'request_id': res['request_id'],
        }, 'A request has been sent to administrator for account creation'), 201
    except Exception as e:
        return error(str(e)), 400


@api.route('/register-admin', methods=['POST', 'GET'])
@validate_request_data(schema=ValidateUserFormRequest)
def register_admin():
    """ API endpoint to register new user in the database (as an admin).

    Request {application/json}:
        email (str): Email of the user to register
        password (str): Password of the user to register
        name (str): Name of the user to register
        surname (str): Surname of the user to register

    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        409:
            success : Boolean error status (False)
            result  : null
            message : Message about error, when user is already registered but wants to register
                        under another account
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
    name = req['name']
    surname = req['surname']

    if not check_mail_format(email):
        return error('Wrong email format'), 400

    if not check_password_format(password):
        return error(
            'Password should be at least 8 character long, with at least one uppercase letter, one lowercase letter and one number'), 400

    if get_user_by_email(email):
        return error('Email already Present. Please log in'), 409
    try:
        # Create unique id for the request
        request_id = 'request_' + str(uuid.uuid4())
        user_requests_table.insert({
            "user_name": name,
            "user_surname": surname,
            "user_email": email,
            "password_hash": set_password_hash(password),
            "user_role": UserRoleType.ADMIN,
            "creation_date": datetime.utcnow().ctime(),
            "request_id": request_id,
            "request_status": UserRequestStatus.NEW
        })
        res = user_requests_table.get(query.request_id == request_id)
        return response({
            'request_id': res['request_id'],
        }, 'A request has been sent to administrator for account creation'), 201
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

    user_db = get_user_by_email(email)
    if not user_db:
        # user account not found
        return error(f'Unrecognized email address {email}. Please register before to log in'), 401

    # Should send back only one item
    user = user_db[0]
    if check_password_hash(password, user['password_hash']):
        additional_claims = {
            "email": user["user_email"],
            "role": user["user_role"]
        }
        access_token = create_access_token(identity=user["user_id"], fresh=True, additional_claims=additional_claims)
        refresh_token = create_refresh_token(identity=user["user_id"], additional_claims=additional_claims)
        resp = response(
            data={
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            message='User successfully logged in')
        return resp, 200
    return error('Please verify your email and/or your password'), 401


@api.route('/token/refresh', methods=['GET'])
@jwt_required(refresh=True)  # only put `refresh` = True here, it means we are accessing api with refresh token instead of access tokens
def refresh_expiring_jwts():
    """ API endpoint for refreshing JWT token. Here we are using "explicit Refreshing", as 
    defined in `jwt-extended` documentation (https://flask-jwt-extended.readthedocs.io/en/stable/refreshing_tokens/).
    """
    jwt = get_jwt()
    additional_claims = {
        "email": jwt["email"],
        "role": jwt["role"]
    }
    access_token = create_access_token(identity=jwt["sub"], additional_claims=additional_claims, fresh=False)
    refresh_token = create_refresh_token(identity=jwt["sub"], additional_claims=additional_claims)
    # TODO: Invalidate old refresh tokens; they should be used only once
    resp = response(
        data={
            "access_token": access_token,
            "refresh_token": refresh_token},
        message='Access token successfully refreshed')
    return resp, 200


@api.route('/protected', methods=['GET'])
@jwt_required()
def protected_test():
    return response('You rock !'), 200


@api.route('/admin', methods=['GET'])
@jwt_required()
@admin_required
def admin_test():
    return response('Only if you are an admin'), 200


@api.route('/token/remove', methods=['POST'])
def logout():
    """ Method used to logout current user.
        It removes the jwt set in cookies
    """
    resp = response(msg='User successfully logged out')
    unset_jwt_cookies(resp)
    return resp, 200

# TODO : Generate secret key server randomly
# TODO : Implement method to retrieve user password
# TODO : Add salt to encrypted passwords
