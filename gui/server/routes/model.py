from fedbiomed.common.constants import ModelApprovalStatus
from gui.server.schemas import ListModelRequest, ApproveModelRequest
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
    select_status = ModelApprovalStatus.str2enum(select_status)
    print(select_status.value)
    res = MODEL_MANAGER.list_models(sort_by=sort_by, 
                                    select_status=select_status,
                                    verbose=False)
    return response(res), 200

@api.route('/model/approve', methods=["POST"])
@validate_request_data(schema=ApproveModelRequest)
def approve_model():
    req = request.json
    model_id = req.get('model_id')
    
    res = MODEL_MANAGER.approve_model(model_id)
    return response(res), 200

