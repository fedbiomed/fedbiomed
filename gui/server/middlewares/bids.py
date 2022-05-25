import os
from fedbiomed.common.data import BIDSController
from fedbiomed.common.exceptions import FedbiomedError
from flask import request, g
from app import app
from utils import error, response


bids_controller = BIDSController()
DATA_PATH_RW = app.config['DATA_PATH_RW']


def read_bids_reference():
    req = request.json
    reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
    index_col = req["index_col"]

    try:
        reference = bids_controller.read_demographics(path=reference_path, index_col=index_col)
    except FedbiomedError:
        return error("Reference demographics should be CSV or TSV"), 400
    except Exception as e:
        return error("Can not read demographics please make sure the file is CSV or TSV and well formatted"), 400

    # Assing it to global `g` state
    g.reference = reference


def validate_bids_root():
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["bids_root"])

    try:
        bids_controller.validate_bids_root_folder(root)
    except FedbiomedError or Exception as e:
        return response({"valid": False, "message": "BIDS root folder is not valid. Please make sure that folder has "
                                                    "been properly structured"}), 400

    bids_controller.root = root
    modalities, _ = bids_controller.modalities()
    g.modalities = modalities


def get_available_subjects():
    """Retries available subjects for BIDS Dataset"""
    req = request.json
    reference = g.reference
    bids_controller.root = os.path.join(DATA_PATH_RW, *req["bids_root"])
    try:
        intersection, missing_folders, missing_entries = \
            bids_controller.available_subjects(subjects_from_index=reference.index)
    except Exception as e:
        return error("Can not get subjects"), 400

    bids_subject = {
        "missing_folders": missing_folders,
        "missing_entries": missing_entries,
        "intersection": intersection,
    }
    g.available_subjects = bids_subject

