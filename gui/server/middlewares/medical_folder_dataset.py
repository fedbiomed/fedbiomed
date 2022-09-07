import os

from app import app
from flask import request, g
from utils import error, response

from fedbiomed.common.data import MedicalFolderController
from fedbiomed.common.exceptions import FedbiomedError

mf_controller = MedicalFolderController()
DATA_PATH_RW = app.config['DATA_PATH_RW']


def read_medical_folder_reference():
    """Reads demographics/reference CSV for BIDS """
    req = request.json
    if not req["reference_csv_path"] or req["reference_csv_path"] is None:
        g.reference = None
        return None

    reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
    index_col = req["index_col"]

    try:
        reference = mf_controller.read_demographics(path=reference_path, index_col=index_col)
    except FedbiomedError:
        return error("Reference demographics should be CSV or TSV"), 400
    except Exception as e:
        return error("Can not read demographics please make sure the file is CSV or TSV and well formatted"), 400

    # Assign MedicalFolder reference to global `g` state
    g.reference = reference


def validate_medical_folder_root():
    """Validates MedicalFolderDataset root folder"""
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["medical_folder_root"])

    try:
        mf_controller.validate_MedicalFolder_root_folder(root)
    except FedbiomedError or Exception as e:
        return error("MedicalFolder root folder is not valid. Please make sure that folder has "
                                                    "been properly structured"), 400

    mf_controller.root = root
    modalities, _ = mf_controller.modalities()
    g.modalities = modalities


def validate_available_subjects():
    """Retries available subjects for MedicalFolder Dataset"""

    if g.reference is None:
        return None

    req = request.json
    reference = g.reference
    mf_controller.root = os.path.join(DATA_PATH_RW, *req["medical_folder_root"])
    try:
        intersection, missing_folders, missing_entries = \
            mf_controller.available_subjects(subjects_from_index=reference.index)
    except Exception as e:
        return error("Can not get subjects"), 400

    if not len(intersection) > 0:
        return response({"valid": False,
                         "message": "Selected column for MedicalFolder subject reference does not correspond "
                                    "any subject folder."}), 200

    mf_subjects = {
        "missing_folders": missing_folders,
        "missing_entries": missing_entries,
        "intersection": intersection,
    }

    g.available_subjects = mf_subjects

