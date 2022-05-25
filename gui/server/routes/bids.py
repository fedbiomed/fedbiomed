import os
import uuid
import re
from flask import jsonify, request, g
from app import app
from db import database
from utils import success, error, validate_json, validate_request_data, response
from schemas import ValidateBIDSReferenceCSV, ValidateBIDSRoot
from . import api
from app import app
from middlewares import middleware, bids

from fedbiomed.common.data import BIDSController
from fedbiomed.common.exceptions import FedbiomedError
bids_controller = BIDSController()
DATA_PATH_RW = app.config['DATA_PATH_RW']


@api.route('/datasets/bids/validate-reference-column', methods=['POST'])
@validate_request_data(schema=ValidateBIDSReferenceCSV)
@middleware(middlewares=[bids.read_bids_reference, bids.get_available_subjects])
def validate_reference_csv_column():
    subjects = g.available_subjects
    if not len(subjects["intersection"]) > 0:
        return response({"valid": False,
                         "message": "Selected column does not correspond any subject folder."}), 200

    return response({"valid": True, "subjects": subjects}), 200


@api.route('/datasets/bids/validate-bids-root', methods=['POST'])
@validate_request_data(schema=ValidateBIDSRoot)
@middleware(middlewares=[bids.validate_bids_root])
def validate_root_path():
    return response(data={"valid": True, "modalities": g.modalities}), 200
