import os
import uuid
import re
from flask import jsonify, request
from app import app
from db import database
from utils import success, error, validate_json, validate_request_data, response
from schemas import ValidateBIDSReferenceCSV, ValidateBIDSRoot

from . import api


@api.route('/datasets/bids/validate-reference-csv', methods=['POST'])
@validate_request_data(schema=ValidateBIDSReferenceCSV)
def validate_reference_csv_column():
    req = request.json

    res = {
        "missing_folders": [],
        "missing_entries": [],
        "complete_subjects": [],
    }
    return response(res), 200


@api.route('/datasets/bids/validate-bids-root', methods=['POST'])
@validate_request_data(schema=ValidateBIDSRoot)
def validate_root_path():
    return response(data={"valid": True, "modalities": ["T1", "T2", "T3"]}), 200
