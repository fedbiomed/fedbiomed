import os
from flask import request, g

from fedbiomed.common.data import (
    MedicalFolderController,
    MedicalFolderDataset,
    DataLoadingPlan,
    MapperBlock,
    MedicalFolderLoadingBlockTypes
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.constants import DatasetTypes
from fedbiomed.node.dataset_manager import DatasetManager

from ..config import config
from ..utils import error, response

mf_controller = MedicalFolderController()
dataset_manager = DatasetManager(config["NODE_DB_PATH"])
DATA_PATH_RW = config['DATA_PATH_RW']


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


def validate_all_modalities():
    """Validates MedicalFolderDataset has subjects with all modalities"""
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["medical_folder_root"])
    modalities = req["modalities"]
    if 'reference_csv_path' in req and req["reference_csv_path"]:
        reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
    else:
        reference_path = None
    index_col = req["index_col"]

    try:
        mf_dataset = MedicalFolderDataset(
            root=root,
            data_modalities=modalities,
            target_modalities=modalities,
            tabular_file=reference_path,
            index_col=index_col
        )
    except FedbiomedError as e:
        return error(f"Cannot instantiate MedicalFolder: {e}"), 400

    if req['dlp_id']:
        try:
            dlp = DataLoadingPlan()
            dlp_and_dlbs_dict = dataset_manager.get_dlp_by_id(req['dlp_id'])
            dlp.deserialize(*dlp_and_dlbs_dict)
        except FedbiomedError as e:
            return error(f"Cannot instantiate data loading plan of MedicalFolder: {e}"), 400
        try:
            mf_dataset.set_dlp(dlp)
        except FedbiomedError as e:
            return error(f"Cannot set data loading plan of medical folder: {e}"), 400

    try:
        subjects = mf_dataset.subjects_has_all_modalities
    except FedbiomedError as e:
        return error(f"Cannot check subjects with all modalities: {e}"), 400

    g.subjects = subjects


def create_dlp():
    """Creates a DLP object from the input values"""
    req = request.json

    try:
        dlb = MapperBlock()
        dlb.map = req["modalities_mapping"]
    except FedbiomedError as e:
        return error(f"Cannot create data loading block for customizations: {e}"), 400

    try:
        dlp = DataLoadingPlan()
        dlp.desc = req["name"]
        dlp.target_dataset_type = DatasetTypes.MEDICAL_FOLDER
        dlp[MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS] = dlb
    except (FedbiomedError, KeyError) as e:
        return error(f"Cannot create data loading plan for customizations: {e}"), 400

    g.dlp = dlp


def load_dlp():
    req = request.json
    dlp = None
    if req['dlp_id'] is not None:
        try:
            dlp = DataLoadingPlan().deserialize(*dataset_manager.get_dlp_by_id(req['dlp_id']))
        except FedbiomedError as e:
            return error(f"Cannot load data loading plan for customizations: {e}"), 400

    g.dlp = dlp


def validate_available_subjects():
    """Retrieves available subjects for MedicalFolder Dataset"""

    if g.reference is None:
        return None

    req = request.json
    reference = g.reference
    mf_controller.root = os.path.join(DATA_PATH_RW, *req["medical_folder_root"])
    try:
        intersection, missing_folders, missing_entries = \
            mf_controller.available_subjects(subjects_from_index=reference.index)
    except Exception as e:
        return error(f"Can not get subjects, error {e}"), 400

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
