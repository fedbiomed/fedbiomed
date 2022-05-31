import os
import uuid
import re

import pandas as pd
from functools import cache
from flask import jsonify, request, g
from db import database
from utils import success, error, validate_json, validate_request_data, response
from schemas import ValidateBIDSReferenceCSV, \
    ValidateBIDSRoot, \
    ValidateBIDSAddRequest, \
    PreviewDatasetRequest
from . import api
from app import app
from middlewares import middleware, bids, common
from fedbiomed.common.data import BIDSController
from cache import cached
bids_controller = BIDSController()
DATA_PATH_RW = app.config['DATA_PATH_RW']


@api.route('/datasets/bids/validate-reference-column', methods=['POST'])
@validate_request_data(schema=ValidateBIDSReferenceCSV)
@middleware(middlewares=[bids.read_bids_reference, bids.validate_available_subjects])
def validate_reference_csv_column():
    """ Validate selected reference CSV and column shows folder names """
    subjects = g.available_subjects
    return response({"valid": True, "subjects": subjects}), 200


@api.route('/datasets/bids/validate-bids-root', methods=['POST'])
@validate_request_data(schema=ValidateBIDSRoot)
@middleware(middlewares=[bids.validate_bids_root])
def validate_root_path():
    """Validates BIDS root path"""
    return response(data={"valid": True, "modalities": g.modalities}), 200


@api.route('/datasets/bids/add', methods=['POST'])
@validate_request_data(schema=ValidateBIDSAddRequest)
@middleware(middlewares=[common.check_tags_already_registered,
                         bids.validate_bids_root,
                         bids.read_bids_reference,
                         bids.validate_available_subjects,
                         bids.create_and_validate_bids_dataset])
def add_bids_dataset():
    """ Adds BIDS dataset into database of NODE """
    req = request.json
    table = database.db().table('_default')
    query = database.query()

    data_path_save = os.path.join(app.config['DATA_PATH_SAVE'], *req['bids_root'])

    # Create unique id for the dataset
    dataset_id = 'dataset_' + str(uuid.uuid4())

    # Get shape
    bids_dataset = g.bids_dataset
    shape = bids_dataset.shape()

    if req["reference_csv_path"] is None:
        dataset_parameters = {}
    else:
        reference_csv = os.path.join(app.config['DATA_PATH_SAVE'], *req["reference_csv_path"])
        dataset_parameters = {"index_col": req["index_col"],
                              "tabular_file": reference_csv}
    try:
        table.insert({
            "name": req["name"],
            "path": data_path_save,
            "data_type": "BIDS",
            "dtypes": [],
            "shape": shape,
            "tags": req['tags'],
            "description": req['desc'],
            "dataset_id": dataset_id,
            "dataset_parameters": dataset_parameters
        })
    except Exception as e:
        return error("yop yop" + str(e)), 400

    # Get saved dataset document
    res = table.get(query.dataset_id == dataset_id)

    return response(data=res), 200


@api.route('/datasets/bids/preview', methods=['POST'])
@validate_request_data(schema=PreviewDatasetRequest)
@cached(key="dataset_id", prefix="bids-preview", timeout=600)
def bids_preview():
    """Gets preview of BIDS dataset by providing a table of subject and available modalities"""
    req = request.json
    table = database.db().table('_default')
    query = database.query()
    dataset = table.get(query.dataset_id == req['dataset_id'])

    # Extract data path where the files are saved in the local GUI repository
    rexp = re.match('^' + app.config['DATA_PATH_SAVE'], dataset['path'])
    data_path = dataset['path'].replace(rexp.group(0), app.config['DATA_PATH_RW'])
    bids_controller.root = data_path

    if "index_col" in dataset["dataset_parameters"]:
        # Extract data path where the files are saved in the local GUI repository
        rexp = re.match('^' + app.config['DATA_PATH_SAVE'], dataset['path'])
        reference_path = dataset["dataset_parameters"]["tabular_file"].replace(rexp.group(0),
                                                                               app.config['DATA_PATH_RW'])

        reference_csv = bids_controller.read_demographics(
            path=reference_path,
            index_col=dataset["dataset_parameters"]["index_col"]
        )

        subject_table = bids_controller.subject_modality_status(index=reference_csv.index)
    else:
        subject_table = bids_controller.subject_modality_status()

    modalities, _ = bids_controller.modalities()

    data = {
        "subject_table": subject_table,
        "modalities": modalities,
    }
    print(data)
    return response(data=data), 200
