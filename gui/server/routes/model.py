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

@api.route('/model/list', methods=['POST'])
@validate_request_data(schema=ListModelRequest)
def list_models():
    req = request.json
    sort_by = req.get('sort_by', None)
    select_status = req.get('select_status', None)
    #print(select_status)
    if select_status is not None:
        select_status = ModelApprovalStatus.str2enum(select_status)
        print(select_status.value)
    res = MODEL_MANAGER.list_models(sort_by=sort_by, 
                                    select_status=select_status,
                                    verbose=False)
    return response(res), 200

@api.route('/model/approve', methods=["POST"])
@validate_request_data(schema=ApproveRejectModelRequest)
def approve_model():
    req = request.json
    model_id = req.get('model_id')
    if model_id is None:
        return error("missing model_id"), 400
    try:
        res = MODEL_MANAGER.approve_model(model_id)
    except FedbiomedModelManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"model {model_id} has been approved"), 200

@api.route('/model/reject', methods=["POST"])
@validate_request_data(schema=ApproveRejectModelRequest)
def reject_model():
    req = request.json
    model_id = req.get('model_id')
    if model_id is None:
        return error("missing model_id"), 400
    try:
        res = MODEL_MANAGER.reject_model(model_id)
    except FedbiomedModelManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"model {model_id} has been rejected"), 200

@api.route('/model/delete', methods=["POST"])
@validate_request_data(schema=DeleteModelRequest)
def delete_model():
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
    req = request.json
    model_path = req.get('model_path')
    if model_path is None:
        return error("missing model_path"), 400
    res = MODEL_MANAGER.get_model_from_database(model_path)
    
    if res is None:
        return error(f"No model with provided model path {model_path} found in database"), 400
    else:
        return response(res), 200
