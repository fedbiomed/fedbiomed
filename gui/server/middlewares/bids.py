import os
from functools import cache
from fedbiomed.common.data import BIDSController
from fedbiomed.common.exceptions import FedbiomedError
from flask import request, g
from app import app
from utils import error, response

bids_controller = BIDSController()
DATA_PATH_RW = app.config['DATA_PATH_RW']


def read_bids_reference():
    req = request.json
    if not req["reference_csv_path"] or req["reference_csv_path"] is None:
        g.reference = None
        return None

    reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
    index_col = req["index_col"]

    try:
        reference = bids_controller.read_demographics(path=reference_path, index_col=index_col)
    except FedbiomedError:
        return error("Reference demographics should be CSV or TSV"), 400
    except Exception as e:
        return error("Can not read demographics please make sure the file is CSV or TSV and well formatted"), 400

    # Assign BIDS reference to global `g` state
    g.reference = reference


def validate_bids_root():
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["bids_root"])

    try:
        bids_controller.validate_bids_root_folder(root)
    except FedbiomedError or Exception as e:
        return error("BIDS root folder is not valid. Please make sure that folder has "
                                                    "been properly structured"), 400

    bids_controller.root = root
    modalities, _ = bids_controller.modalities()
    g.modalities = modalities


def validate_available_subjects():
    """Retries available subjects for BIDS Dataset"""

    if g.reference is None:
        return None

    req = request.json
    reference = g.reference
    bids_controller.root = os.path.join(DATA_PATH_RW, *req["bids_root"])
    try:
        intersection, missing_folders, missing_entries = \
            bids_controller.available_subjects(subjects_from_index=reference.index)
    except Exception as e:
        return error("Can not get subjects"), 400

    if not len(intersection) > 0:
        return response({"valid": False,
                         "message": "Selected column for BIDS subject reference does not correspond "
                                    "any subject folder."}), 200

    bids_subject = {
        "missing_folders": missing_folders,
        "missing_entries": missing_entries,
        "intersection": intersection,
    }

    g.available_subjects = bids_subject


def create_and_validate_bids_dataset():
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["bids_root"])
    if req.get("reference_csv_path", None) is None:
        reference_path = None
        index_col = None
    else:
        reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
        index_col = req["index_col"]

    try:
        bids_controller.root = root
        bids = bids_controller.load_bids(tabular_file=reference_path, index_col=index_col)
    except FedbiomedError as e:
        return error(f"Can not add BIDS dataset. The error message is '{e}'"), 400
    except Exception as e:
        print("Unexpected error: ", e)
        return error(f"Unexpected error while validating BIDS dataset. Please contact to system provider.'"), 400

    try:
        x = bids[0]
    except Exception as e:
        return error("Error while validating BIDS dataset. Pleas emake sure BIDS has been formatted as expected.")

    g.bids_dataset = bids


