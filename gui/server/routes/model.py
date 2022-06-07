from datetime import datetime
from fedbiomed.common.constants import ModelApprovalStatus
from fedbiomed.common.exceptions import FedbiomedModelManagerError
from gui.server.schemas import DeleteModelRequest, ListModelRequest, ApproveRejectModelRequest, ModelPreviewRequest
from . import api

from flask import jsonify, request
from app import app
from db import database
from utils import success, error, validate_json, validate_request_data, response

from fedbiomed.node.model_manager import ModelManager

MODEL_MANAGER = ModelManager()

TIME_OF_LAST_CALL = datetime.now()


@api.route('/model/list', methods=['POST'])
@validate_request_data(schema=ListModelRequest)
def list_models():
    """API endpoint for listing model contained in database

    ---
    Request {application/json}:
            sort_by (str): sort result along one column in the database
            select_status (str): filter result by model statuses {Pending, Rejected, Approved}
            
    Response {application/json}:
        400:
            success   : Boolean error status (False)
            result  : null
            message : Message about error. 

        200:
            success : Boolean value indicates that the request is success
            result  : list of models, sorted or filtered depending on [`sort_by`] or [`select_status`]
                      arguments
            message : null  
    """
    req = request.json
    sort_by = req.get('sort_by', None)
    select_status = req.get('select_status', None)

    if select_status is not None:
        select_status = ModelApprovalStatus.str2enum(select_status)

    res = MODEL_MANAGER.list_models(sort_by=sort_by, 
                                    select_status=select_status,
                                    verbose=False)
    return response(res), 200


@api.route('/model/approve', methods=["POST"])
@validate_request_data(schema=ApproveRejectModelRequest)
def approve_model():
    """API endpoint for approving model

    ---

    Request {application/json}:
            model_id  : model id that is approved (Required)

    Response {application/json}:
        400:
            success: False
            result: null
            message: Error when approving model
        200:
            success: True
            result: null
            message: model has been approved
    """
    req = request.json
    model_id = req.get('model_id')
    model_note = req.get('notes', None)

    if model_id is None:
        return error("missing model_id"), 400
    try:
        res = MODEL_MANAGER.approve_model(model_id, model_note)
    except FedbiomedModelManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"model {model_id} has been approved"), 200


@api.route('/model/reject', methods=["POST"])
@validate_request_data(schema=ApproveRejectModelRequest)
def reject_model():
    """API endpoint for rejecting model

    ---

    Request {application/json}:
            model_id  : model id that is rejected(required)

    Response {application/json}:
        400:
            success: False
            result: null
            message: Error when rejecting model
        200:
            success: True
            result: null
            message: model has been rejected
    """
    req = request.json
    model_id = req.get('model_id')
    model_note = req.get('notes', None)

    if model_id is None:
        return error("missing model_id"), 400
    try:
        res = MODEL_MANAGER.reject_model(model_id, extra_notes=model_note)
    except FedbiomedModelManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"model {model_id} has been rejected"), 200


@api.route('/model/delete', methods=["POST"])
@validate_request_data(schema=DeleteModelRequest)
def delete_model():
    """API endpoint for deleting model
    
    ----
    Request {application/json}:
            model_id  : model id that should be deleted (required)

    Response {application/json}:
        400:
            success: False
            result: null
            message: Error when deleting model
        200:
            success: True
            result: null
            message: model has been deleted
    """
    req = request.json
    model_id = req.get('model_id')
    if model_id is None:
        return error("missing model_id"), 400
    try:
        res = MODEL_MANAGER.delete_model(model_id)
    except FedbiomedModelManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"model {model_id} has been deleted"), 200


@api.route('/model/preview', methods=["POST"])
@validate_request_data(schema=ModelPreviewRequest)
def preview_model():
    """API endpoint for getting a specific model entry through [`model_id`]
    ---
    
    Request {application/json}:
            model_id  : model id to look for in the database

    Response {application/json}:
        400:
            success: False
            result: null
            message: Bad Request
        200:
            success: True
            result: model entry
            message: null
    """
    req = request.json

    model_id = req.get('model_id')
    
    try:
        res = MODEL_MANAGER.get_model_by_id(model_id, content=True)
    except FedbiomedModelManagerError as fed_err:
        return error(f"Bad request. Details: {fed_err}"), 400
    if res is None:
        return error(f"No model with provided model id {model_id} found in database"), 400
    else:
        return response(res), 200


@api.route('/model/update-list-model', methods=["GET"])
def update_list_model():
    """API endpoint for getting only newly added model in database
    (avoid returning model list each time for performance sake).
    Returns either models that have been added after the last call to
    `update_list_model` or modified (through `date_last_action` field) or 
    which have missing dates
    ---
    Response {application/json}:
        200:
            success: True
            result: list of newly added models
            message: null
    """
    global TIME_OF_LAST_CALL
    print("LAST CALL", TIME_OF_LAST_CALL)
    models = MODEL_MANAGER.list_models(sort_by='date_last_action', verbose=False)
    updated_models = []
    # we sort list from the more recent updated models to the less recent ones (list_model
    # returns from the older to newer)
    for model in models[::-1]:
        date = model.get('date_last_action', datetime.now())
        if isinstance(date, str):
            date = datetime.strptime(date, "%d-%m-%Y %H:%M:%S.%f")
        if date is None or TIME_OF_LAST_CALL < date:
            # Append to list only recent models
            updated_models.append(model)

    TIME_OF_LAST_CALL = datetime.now()
    
    return response(updated_models), 200
