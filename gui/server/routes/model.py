from datetime import datetime
from flask import request

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager

from . import api
from ..config import config
from ..schemas import (
    DeleteTrainingPlanRequest,
    ListTrainingPlanRequest,
    ApproveRejectTrainingPlanRequest,
    TrainingPlanPreviewRequest
)
from ..utils import success, error, validate_request_data, response


TP_SECURITY_MANAGER = TrainingPlanSecurityManager(
    db=config["NODE_DB_PATH"],
    node_id=config["ID"],
    hashing=config.node_config.get('security', 'hashing_algorithm'),
    tp_approval=config.node_config.getbool('security', 'training_plan_approval')
)

TIME_OF_LAST_CALL = datetime.now()


@api.route('/training-plan/list', methods=['POST'])
@validate_request_data(schema=ListTrainingPlanRequest)
def list_training_plans():
    """API endpoint for listing training plan contained in database

    ---
    Request {application/json}:
            sort_by (str): sort result along one column in the database
            select_status (str): filter result by training plan statuses {Pending, Rejected, Approved}

    Response {application/json}:
        400:
            success   : Boolean error status (False)
            result  : null
            message : Message about error.

        200:
            success : Boolean value indicates that the request is success
            result  : list of training plans, sorted or filtered depending on [`sort_by`] or [`select_status`]
                      arguments
            message : null
    """
    req = request.json
    sort_by = req.get('sort_by', None)
    select_status = req.get('select_status', None)
    search = req.get('search', None)

    if select_status is not None:
        select_status = [TrainingPlanApprovalStatus.str2enum(select_status)]

    res = TP_SECURITY_MANAGER.list_training_plans(sort_by=sort_by,
                                                  select_status=select_status,
                                                  verbose=False,
                                                  search=search)
    return response(res), 200


@api.route('/training-plan/approve', methods=["POST"])
@validate_request_data(schema=ApproveRejectTrainingPlanRequest)
def approve_training_plan():
    """API endpoint for approving training plan

    ---

    Request {application/json}:
            training_plan_id  : training plan id that is approved (Required)

    Response {application/json}:
        400:
            success: False
            result: null
            message: Error when approving training plan
        200:
            success: True
            result: null
            message: training plan has been approved
    """
    req = request.json
    training_plan_id = req.get('training_plan_id')
    training_plan_note = req.get('notes', None)

    if training_plan_id is None:
        return error("missing training_plan_id"), 400
    try:
        res = TP_SECURITY_MANAGER.approve_training_plan(training_plan_id, training_plan_note)
    except FedbiomedTrainingPlanSecurityManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"training_plan {training_plan_id} has been approved"), 200


@api.route('/training-plan/reject', methods=["POST"])
@validate_request_data(schema=ApproveRejectTrainingPlanRequest)
def reject_training_plan():
    """API endpoint for rejecting training plan

    ---

    Request {application/json}:
            training_plan_id  : training plan id that is rejected(required)

    Response {application/json}:
        400:
            success: False
            result: null
            message: Error when rejecting training plan
        200:
            success: True
            result: null
            message: training plan has been rejected
    """
    req = request.json
    training_plan_id = req.get('training_plan_id')
    training_plan_note = req.get('notes', None)

    if training_plan_id is None:
        return error("missing training_plan_id"), 400
    try:
        res = TP_SECURITY_MANAGER.reject_training_plan(training_plan_id, extra_notes=training_plan_note)
    except FedbiomedTrainingPlanSecurityManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"Training plan {training_plan_id} has been rejected"), 200


@api.route('/training-plan/delete', methods=["POST"])
@validate_request_data(schema=DeleteTrainingPlanRequest)
def delete_training_plan():
    """API endpoint for deleting training plan

    ----
    Request {application/json}:
            training_plan_id  : training plan id that should be deleted (required)

    Response {application/json}:
        400:
            success: False
            result: null
            message: Error when deleting training plan
        200:
            success: True
            result: null
            message: training plan has been deleted
    """
    req = request.json
    training_plan_id = req.get('training_plan_id')
    if training_plan_id is None:
        return error("missing training_plan_id"), 400
    try:
        res = TP_SECURITY_MANAGER.delete_training_plan(training_plan_id)
    except FedbiomedTrainingPlanSecurityManagerError as fed_err:
        return error(str(fed_err)), 400
    return success(f"training_plan {training_plan_id} has been deleted"), 200


@api.route('/training-plan/preview', methods=["POST"])
@validate_request_data(schema=TrainingPlanPreviewRequest)
def preview_training_plan():
    """API endpoint for getting a specific training plan entry through [`training_plan_id`]
    ---

    Request {application/json}:
            training_plan_id  : training plan id to look for in the database

    Response {application/json}:
        400:
            success: False
            result: null
            message: Bad Request
        200:
            success: True
            result: training plan entry
            message: null
    """
    req = request.json

    training_plan_id = req.get('training_plan_id')

    try:
        res = TP_SECURITY_MANAGER.get_training_plan_by_id(training_plan_id)
    except FedbiomedTrainingPlanSecurityManagerError as fed_err:
        return error(f"Bad request. Details: {fed_err}"), 400
    if res is None:
        return error(f"No training plan with provided training plan id {training_plan_id} found in database"), 400
    else:
        return response(res), 200
