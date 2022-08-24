import uuid

from fedbiomed.common.constants import UserRequestStatus
from gui.server.routes import admin_required
from . import api
from utils import success, error, validate_request_data, response

from flask import request
from flask_jwt_extended import jwt_required

from db import user_database
from gui.server.schemas import ValidateAdminRequestAction

user_table = user_database.table('Users')
user_requests_table = user_database.table('Requests')
query = user_database.query()


@api.route('/admin/requests/list', methods=['GET'])
@jwt_required()
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
    req = request.json
    search = req.get('search', None)

    if search is not None and search != "":
        res = user_requests_table.search(query.name.search(search + '+') | query.description.search(search + '+'))
    else:
        try:
            res = user_requests_table.all()
        except Exception as e:
            return error(str(e)), 400

    return response(res), 200


@api.route('/admin/requests/approve', methods=['POST'])
@jwt_required()
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
            "creation_date": user_request["creation_date"],
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
@jwt_required()
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
