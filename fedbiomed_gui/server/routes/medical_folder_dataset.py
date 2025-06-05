import os
import re

from flask import request, g

from fedbiomed.common.data import MedicalFolderController
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager import DatasetManager

from .api import api
from ..config import config
from ..cache import cached
from ..db import node_database
from ..middlewares import middleware, medical_folder_dataset, common
from ..schemas import (
    ValidateMedicalFolderReferenceCSV,
    ValidateMedicalFolderRoot,
    ValidateSubjectsHasAllModalities,
    ValidateMedicalFolderAddRequest,
    ValidateDataLoadingPlanAddRequest,
    ValidateDataLoadingPlanDeleteRequest,
    PreviewDatasetRequest
)
from ..utils import error, validate_request_data, response

dataset_manager = DatasetManager(config["NODE_DB_PATH"])

# Medical Folder Controller
mf_controller = MedicalFolderController()

# Path to write and read the datafiles
DATA_PATH_RW = config['DATA_PATH_RW']

# Database table (default datasets table of TinyDB) and query object
table = node_database.table_datasets()
query = node_database.query()


@api.route('/datasets/medical-folder-dataset/validate-reference-column', methods=['POST'])
@validate_request_data(schema=ValidateMedicalFolderReferenceCSV)
@middleware(middlewares=[medical_folder_dataset.read_medical_folder_reference,
                         medical_folder_dataset.validate_available_subjects])
def validate_reference_csv_column():
    """ Validate selected reference CSV and column shows folder names """
    subjects = g.available_subjects
    return response({"valid": True, "subjects": subjects}), 200


@api.route('/datasets/medical-folder-dataset/validate-root', methods=['POST'])
@validate_request_data(schema=ValidateMedicalFolderRoot)
@middleware(middlewares=[medical_folder_dataset.validate_medical_folder_root])
def validate_root_path():
    """Validates MedicalFolder Dataset root path"""
    return response(data={"valid": True, "modalities": g.modalities}), 200


@api.route('/datasets/medical-folder-dataset/validate-all-modalities', methods=['POST'])
@validate_request_data(schema=ValidateSubjectsHasAllModalities)
@middleware(middlewares=[medical_folder_dataset.validate_all_modalities])
def validate_subjects_has_all_modalities():
    """Validates MedicalFolder Dataset has subjects with all modalities"""
    return response(data={"valid": True, "subjects": g.subjects}), 200


@api.route('/datasets/medical-folder-dataset/add', methods=['POST'])
@validate_request_data(schema=ValidateMedicalFolderAddRequest)
@middleware(middlewares=[common.check_tags_already_registered,
                         medical_folder_dataset.load_dlp,
                         medical_folder_dataset.validate_medical_folder_root,
                         medical_folder_dataset.read_medical_folder_reference,
                         medical_folder_dataset.validate_available_subjects])
def add_medical_folder_dataset():
    """ Adds MedicalFolder dataset into database of NODE """

    # Request object as JSON
    req = request.json

    data_path_save = os.path.join(config['DATA_PATH_SAVE'], *req['medical_folder_root'])

    if req["reference_csv_path"] is None:
        dataset_parameters = {}
    else:
        reference_csv = os.path.join(config['DATA_PATH_SAVE'], *req["reference_csv_path"])
        dataset_parameters = {"index_col": req["index_col"],
                              "tabular_file": reference_csv}
    try:
        dataset_id = dataset_manager.add_database(
            name=req["name"],
            data_type="medical-folder",
            tags=req['tags'],
            description=req['desc'],
            path=data_path_save,
            dataset_parameters=dataset_parameters,
            data_loading_plan=g.dlp,
            save_dlp=False)
    except FedbiomedError as e:
        return error(str(e)), 400
    except Exception as e:
        return error("Unexpected error: " + str(e)), 400

    # Get saved dataset document
    res = table.get(query.dataset_id == dataset_id)
    if not res:
        return error("Medical Folder Dataset is not properly deployed. "
                     "Please try again."), 400

    return response(data=res), 200


@api.route('/datasets/medical-folder-dataset/add-dlp', methods=['POST'])
@validate_request_data(schema=ValidateDataLoadingPlanAddRequest)
@middleware(middlewares=[medical_folder_dataset.create_dlp])
def add_data_loading_plan():
    """Adds DataLoadingPlan into database of NODE """

    try:
        dlp_id = dataset_manager.save_data_loading_plan(g.dlp)
    except FedbiomedError as e:
        return error(f"Cannot save data loading plan for customizations: {e}"), 400
    if dlp_id is None:
        return error("Cannot save data loading plan for customizations: no DLP id"), 400

    return response(data=dlp_id), 200


@api.route('/datasets/medical-folder-dataset/delete-dlp', methods=['POST'])
@validate_request_data(schema=ValidateDataLoadingPlanDeleteRequest)
def remove_data_loading_plan():
    """Remove DataLoadingPlan from database of NODE """
    # Request object as JSON
    req = request.json

    try:
        dataset_manager.remove_dlp_by_id(req['dlp_id'], True)
    except FedbiomedError as e:
        return error(f"Cannot remove data loading plan for customizations: {e}"), 400

    return response(data=True), 200


@api.route('/datasets/medical-folder-dataset/preview', methods=['POST'])
@validate_request_data(schema=PreviewDatasetRequest)
@cached(key="dataset_id", prefix="medical_folder_dataset-preview", timeout=600)
def medical_folder_preview():
    """Gets preview of MedicalFolder dataset by providing a table of subject and available modalities"""

    # Request object as JSON
    req = request.json

    dataset = table.get(query.dataset_id == req['dataset_id'])

    # Extract data path where the files are saved in the local GUI repository
    rexp = re.match('^' + config['DATA_PATH_SAVE'], dataset['path'])
    data_path = dataset['path'].replace(rexp.group(0), config['DATA_PATH_RW'])
    mf_controller.root = data_path

    if "index_col" in dataset["dataset_parameters"]:
        # Extract data path where the files are saved in the local GUI repository
        rexp = re.match('^' + config['DATA_PATH_SAVE'], dataset['path'])
        reference_path = dataset["dataset_parameters"]["tabular_file"].replace(rexp.group(0),
                                                                               config['DATA_PATH_RW'])

        reference_csv = mf_controller.read_demographics(
            path=reference_path,
            index_col=dataset["dataset_parameters"]["index_col"]
        )

        subject_table = mf_controller.subject_modality_status(index=reference_csv.index)
    else:
        subject_table = mf_controller.subject_modality_status()

    modalities, _ = mf_controller.modalities()

    data = {
        "subject_table": subject_table,
        "modalities": modalities,
    }
    return response(data=data), 200


@api.route('/datasets/medical-folder-dataset/default-modalities', methods=['GET'])
def get_default_modalities():
    formatted_modalities = [{'value': name, 'label': name} for name in MedicalFolderController.default_modality_names]
    return response(data={'default_modalities': formatted_modalities}), 200
