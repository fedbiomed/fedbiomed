import os

from flask import g, request

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.dataloadingplan import (
    DataLoadingPlan,
    MapperBlock,
)
from fedbiomed.common.dataset_controller import (
    MedicalFolderController,
    MedicalFolderLoadingBlockTypes,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager import DatasetManager

from ..config import config
from ..utils import error, response

dataset_manager = DatasetManager(config["NODE_DB_PATH"])
DATA_PATH_RW = config["DATA_PATH_RW"]


def read_medical_folder_reference():
    """Reads demographics/reference CSV for BIDS"""
    req = request.json
    if not req["reference_csv_path"] or req["reference_csv_path"] is None:
        g.reference = None
        return None

    reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
    index_col = req["index_col"]

    try:
        reference = MedicalFolderController.read_demographics(
            tabular_file=reference_path, index_col=index_col
        )
    except FedbiomedError:
        return error("Reference demographics should be CSV or TSV"), 400
    except Exception as e:
        print("Exception")
        return error(
            f"Can not read demographics please make sure the file is CSV or TSV and well formatted {e}"
        ), 400

    print("After read")
    # Assign MedicalFolder reference to global `g` state
    g.reference = reference


def validate_medical_folder_root():
    """Validates MedicalFolderDataset root folder"""
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["medical_folder_root"])

    try:
        mf_controller = MedicalFolderController(root=root)
    except (FedbiomedError, Exception):
        return error(
            "MedicalFolder root folder is not valid. Please make sure that folder has "
            "been properly structured"
        ), 400

    g.modalities = mf_controller.modalities


def validate_all_modalities():
    """Validates MedicalFolderDataset has subjects with all modalities"""
    req = request.json
    root = os.path.join(DATA_PATH_RW, *req["medical_folder_root"])
    if "reference_csv_path" in req and req["reference_csv_path"]:
        reference_path = os.path.join(DATA_PATH_RW, *req["reference_csv_path"])
    else:
        reference_path = None
    index_col = req["index_col"]

    try:
        mf_controller = MedicalFolderController(
            root=root,
            tabular_file=reference_path,
            index_col=index_col,
            validate=False if req["dlp_id"] else True,
        )
    except FedbiomedError as e:
        return error(f"Cannot instantiate MedicalFolder: {e}"), 400

    if req["dlp_id"]:
        try:
            dlp = DataLoadingPlan()
            dlp_and_dlbs_dict = dataset_manager.get_dlp_by_id(req["dlp_id"])
            dlp.deserialize(*dlp_and_dlbs_dict)
        except FedbiomedError as e:
            return error(
                f"Cannot instantiate data loading plan of MedicalFolder: {e}"
            ), 400
        try:
            mf_controller.set_dlp(dlp)
        except FedbiomedError as e:
            return error(f"Cannot set data loading plan of medical folder: {e}"), 400

    try:
        subjects = mf_controller.subjects
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
    if req["dlp_id"] is not None:
        try:
            dlp = DataLoadingPlan().deserialize(
                *dataset_manager.get_dlp_by_id(req["dlp_id"])
            )
        except FedbiomedError as e:
            return error(f"Cannot load data loading plan for customizations: {e}"), 400

    g.dlp = dlp


def validate_available_subjects():
    """Retrieves available subjects for MedicalFolder Dataset"""

    if g.reference is None:
        return None

    req = request.json
    reference = g.reference
    mf_controller = MedicalFolderController(
        root=os.path.join(DATA_PATH_RW, *req["medical_folder_root"])
    )
    try:
        mf_subjects = mf_controller.available_subjects(
            subjects_from_index=reference.index
        )
    except Exception as e:
        return error(f"Can not get subjects, error {e}"), 400

    if len(mf_subjects["intersection"]) == 0:
        return response(
            {
                "valid": False,
                "message": "Selected column for MedicalFolder subject reference does not correspond "
                "any subject folder.",
            }
        ), 200

    g.available_subjects = mf_subjects
