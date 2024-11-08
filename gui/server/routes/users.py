import uuid
import secrets
import string
from tinydb import where
from flask import request
from flask_jwt_extended import jwt_required, get_jwt
from datetime import datetime

from fedbiomed.common.constants import UserRequestStatus
from fedbiomed.common.constants import UserRoleType

from ..db import user_database
from ..schemas import (
    ValidateAdminRequestAction,
    ValidateUserFormRequest,
    ValidateUserRemoveRequest,
    ValidateUserChangeRoleRequest
)
from ..middlewares.auth_validation import validate_email_register, validate_password
from ..middlewares import middleware
from ..helpers.auth_helpers import set_password_hash, admin_required
from ..utils import error, validate_request_data, response
from .api import api

user_table = user_database.table('Users')
user_requests_table = user_database.table('Requests')
query = user_database.query()


@api.route('/admin/users/list', methods=['GET'])
@admin_required
def list_users():
    """
        List of registered users in GUI DB

        Request.GET {None}:
            - No request data

        Response {application/json}:
            400:
                success  : Boolean error status (False)
                result  : null
                message : Message about error. Can be validation error or
                          error from TinyDBt
            200:
                success: Boolean value indicates that the request is success
                result: List of user objects
                endpoint: API endpoint
                message: The message for response
        """

    try:
        users = user_table.all()
    except Exception as e:
        return error(f'Error while getting users {e}'), 400

    return response(users), 200


@api.route('/admin/users/create', methods=['POST'])
@admin_required
@validate_request_data(schema=ValidateUserFormRequest)
@middleware(middlewares=[validate_email_register, validate_password])
def create_user():
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

    email = req['email']
    password = req['password']
    name = req['name']
    surname = req['surname']

    if req['confirm'] != req['password']:
        return error(
            'Password confirmation does not match to the password'), 400

    try:
        # Create unique id for the request
        user_id = 'user_' + str(uuid.uuid4())
        user_table.insert({
            "user_name": name,
            "user_surname": surname,
            "user_email": email,
            "password_hash": set_password_hash(password),
            "user_role": UserRoleType.USER,
            "creation_date": datetime.now().isoformat(),
            "user_id": user_id,
        })
    except Exception as e:
        return error(str(e)), 400

    try:
        res = user_table.get(query.user_id == user_id)
    except Exception as e:
        return error(f'Error while validating user removal status. User might be removed but this '
                     f'status is not validated by the API. Please check user list to make sure it is deleted. '
                     f'Error {e}'), 400

    if res:
        return response(res, 'User has been successfully registered!'), 201
    else:
        return error('Unexpected error, please try again later'), 400


@api.route('/admin/users/remove', methods=['DELETE'])
@admin_required
@validate_request_data(schema=ValidateUserRemoveRequest)
def remove_user():
    """ API endpoint to remove user account (as an admin).

    Request {application/json}:
        user_id (str): ID of the user that will be removed


    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        200:
            success : Boolean value indicates that the request is success
            result  : null
            message : The message for response
    """

    req = request.json
    user_id = req["user_id"]

    user = get_jwt()

    # User can not remove his account
    if user["sub"] == user_id:
        return error('Cannot remove your own account.'), 400

    try:
        user_table.remove(where('user_id') == user_id)
    except Exception as e:
        return error(str(e)), 400

    try:
        res = user_table.get(query.user_id == user_id)
    except Exception as e:
        return error(f'Error while validating user removal status. User might be removed but this '
                     f'status is not validated by the API. Please check user list to make sure it is deleted. '
                     f'Error {e}'), 400

    if not res:
        return response({"user_id": user_id}, 'User has been successfully deleted!'), 200
    else:
        return error('User is not removed. Please try again or contact to system manager.'), 400


@api.route('/admin/users/reset-password', methods=['PATCH'])
@admin_required
@validate_request_data(schema=ValidateUserRemoveRequest)
def reset_user_password():
    """ API endpoint to reset user password (as an admin).

    Request {application/json}:
        user_id (str): ID of the user whose password will be regenerated


    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        200:
            success : Boolean value indicates that the request is success
            result  : null
            message : The message for response
    """

    req = request.json
    user_id = req["user_id"]

    user = get_jwt()

    # User can not remove his account
    if user["sub"] == user_id:
        return error('As admin you can not remove your own account.'), 400

    # Auto generated password
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for _ in range(12))
    password_hash = set_password_hash(password)

    try:
        res = user_table.update({"password_hash": password_hash}, query.user_id == user_id)
    except Exception as e:
        return error(str(e)), 400

    if res:
        user = user_table.get(query.user_id == user_id)
        return response({"password": password, "email": user["user_email"]}, 'User password has been '
                                                                             'successfully updated.'), 200
    else:
        return error('Can not update user password. User may not be existing'), 400


@api.route('/admin/users/change-role', methods=['PATCH'])
@admin_required
@validate_request_data(schema=ValidateUserChangeRoleRequest)
def change_user_role():
    """ API endpoint to change user role (as an admin).

    Request {application/json}:
        user_id (str): ID of the user that will be removed
        role (int): role

    Response {application/json}:
        400:
            error   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        200:
            success : Boolean value indicates that the request is success
            result  : null
            message : The message for response
    """

    req = request.json
    user_id = req["user_id"]
    role = req["role"]
    user = get_jwt()

    # User can not remove his account
    if user["sub"] == user_id:
        return error('You can not change your own role'), 400

    try:
        res = user_table.update({"user_role": role}, query.user_id == user_id)
    except Exception as e:
        return error(str(e)), 400

    if res:
        user = user_table.get(query.user_id == user_id)
        return response({"role": role, "email": user["user_email"]}, 'User role has been successfully changed'), 200
    else:
        return error('Can not update user role. User may not be existing'), 400


@api.route('/admin/requests/list', methods=['GET'])
@admin_required
def list_requests():
    """
        List user registration requests saved into GUI DB

        Request.GET {None}:
            - No request data

        Response {application/json}:
            400:
                success   : Boolean error status (False)
                result  : null
                message : Message about error. Can be validation error or
                          error from TinyDBt
            200:
                success: Boolean value indicates that the request is success
                result: List of request objects
                endpoint: API endpoint
                message: The message for response
        """
    # req = request.json
    # print(req)
    # search = req.get('search', None)
    #
    # if search is not None and search != "":
    #     res = user_requests_table.search(query.name.search(search + '+') | query.description.search(search + '+'))
    # else:
    try:
        res = user_requests_table.all()
    except Exception as e:
        return error(str(e)), 400

    return response(res), 200


@api.route('/admin/requests/approve', methods=['POST'])
@admin_required
@validate_request_data(schema=ValidateAdminRequestAction)
def approve_user_request():
    """ API endpoint to approve new user request and register
    user in the database (as a simple user).

    Request {application/json}:
        request_id (str): id of the request to approve

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

    try:
        request_id = request.json['request_id']
        user_request = user_requests_table.get(query.request_id == request_id)
        if not user_request:
            return error(f'Request with id {request_id} not found'), 400
        user_id = 'user_' + str(uuid.uuid4())
        user_table.insert({
            "user_name": user_request["user_name"],
            "user_surname": user_request["user_surname"],
            "user_email": user_request["user_email"],
            "password_hash": user_request["password_hash"],
            "user_role": user_request["user_role"],
            "creation_date": datetime.utcnow().ctime(),
            "user_id": user_id
        })
        res = user_table.get(query.user_id == user_id)
        user_requests_table.remove(query.request_id == request_id)
        return response({
            'user_id': res['user_id'],
            'user_email': res['user_email']
        }, 'Request successfully approved'), 201
    except Exception as e:
        return error(str(e)), 400


@api.route('/admin/requests/reject', methods=['POST'])
@admin_required
@validate_request_data(schema=ValidateAdminRequestAction)
def reject_user_request():
    """ API endpoint to reject new user request.

    Request {application/json}:
        request_id (str): id of the request to approve

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
    try:
        request_id = request.json['request_id']
        user_request = user_requests_table.get(query.request_id == request_id)
        if not user_request:
            return error(f'Request with id {request_id} not found'), 400
        user_requests_table.update({
            "request_status": UserRequestStatus.REJECTED
        }, query.request_id == request_id)
        res = user_requests_table.get(query.request_id == request_id)
        return response(res), 200
    except Exception as e:
        return error(str(e)), 400

# TODO: Find a way to notify the user about rejection or acceptation of his/her request


