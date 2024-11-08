import re
import uuid
from datetime import datetime
from hashlib import sha512
from flask import request
from flask_jwt_extended import (
    jwt_required,
    create_access_token,
    create_refresh_token,
    unset_jwt_cookies,
    get_jwt
)

from fedbiomed.common.constants import UserRoleType, UserRequestStatus

from ..helpers.auth_helpers import  (
    get_user_by_email,
    set_password_hash,
    check_password_hash
)

from ..utils import error, response
from ..schemas import ValidateUserFormRequest, ValidateLoginRequest
from ..middlewares.auth_validation import validate_email_register, validate_password
from ..middlewares import middleware
from ..utils import validate_request_data
from ..db import user_database
from .api import api, auth

user_table = user_database.table('Users')
user_requests_table = user_database.table('Requests')
query = user_database.query()


@api.route('/update-password', methods=['POST'])
@validate_request_data(schema=ValidateUserFormRequest)
@middleware(middlewares=[validate_password])
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
    email, password, old_password = req['email'], req['password'], req['old_password']
    decoded_json = get_jwt()

    if decoded_json['email'] != email:
        # TODO: allow also operation if user's role is admin
        return error('Error invalid user id'), 400

    user_name = get_user_by_email(email)

    if not user_name:
        return error('Invalid operation: User does not belong to database'), 400

    try:

        res = user_table.get(query.user_email == email)
        if not check_password_hash(old_password, res['password_hash']):
            # check that old password provided is correct
            return error("Incorrect old password"), 400
        user_table.update({
            "password_hash": set_password_hash(password)
        }, query.user_email == decoded_json['email'])
        res = user_table.get(query.user_email == email)

        return response({
            'user_id': res['user_id'],
            'user_email': email}, 'User password successfully updated'), 200

    except Exception as e:
        return error(str(e)), 400


@auth.route('/register', methods=['POST', 'GET'])
@validate_request_data(schema=ValidateUserFormRequest)
@middleware(middlewares=[validate_email_register, validate_password])
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

    email = req['email']
    password = req['password']
    name = req['name']
    surname = req['surname']

    if req['confirm'] != req['password']:
        return error(
            'Password confirmation does not match to the password'), 400

    try:
        # Create unique id for the request
        request_id = 'request_' + str(uuid.uuid4())
        user_requests_table.insert({
            "user_name": name,
            "user_surname": surname,
            "user_email": email,
            "password_hash": set_password_hash(password),
            "user_role": UserRoleType.USER,
            "creation_date": datetime.now().isoformat(),
            "request_id": request_id,
            "request_status": UserRequestStatus.NEW
        })
    except Exception as e:
        return error(str(e)), 400

    res = user_requests_table.get(query.request_id == request_id)

    return response({
        'request_id': res['request_id'],
    }, 'A request has been sent to administrator for account creation'), 201


@api.route('/token/auth', methods=['GET'])
def auto_auth():
    user_info = get_jwt()

    return response(user_info), 200


@auth.route('/token/login', methods=['POST'])
@validate_request_data(schema=ValidateLoginRequest)
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

    email = req['email']
    password = req['password']

    user = get_user_by_email(email)

    if not user:
        # user account not found
        return error(f'Unrecognized email address {email}. Please register before to log in'), 401

    # Should send back only one item
    if check_password_hash(password, user['password_hash']):
        additional_claims = {
            "email": user["user_email"],
            "role": user["user_role"],
            "name": user.get("user_name", "No-name"),
            "surname": user.get("user_surname", "No-name")
        }
        access_token = create_access_token(identity=user["user_id"], fresh=True, additional_claims=additional_claims)
        refresh_token = create_refresh_token(identity=user["user_id"], additional_claims=additional_claims)

        # Update last login
        user_table.update({'last_login': datetime.now().isoformat()}, query.user_id == user['user_id'])

        resp = response(
            data={
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            message='User successfully logged in')
        return resp, 200


    return error('Please verify your email and/or your password'), 401


@auth.route('/token/refresh', methods=['GET'])
@jwt_required(refresh=True)
# `refresh` = True here, it means accessing api with refresh token instead of access tokens
def refresh_expiring_jwts():
    """ API endpoint for refreshing JWT token.

    Here we are using "explicit Refreshing", as defined in `jwt-extended` documentation
    (https://flask-jwt-extended.readthedocs.io/en/stable/refreshing_tokens/).
    """
    jwt = get_jwt()
    additional_claims = {
        "email": jwt["email"],
        "name": jwt["name"],
        "surname": jwt["surname"],
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
