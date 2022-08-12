import os
import uuid
import re

from cache import cached
from flask import jsonify, request, g
from db import database
from . import api
from app import app
from middlewares import middleware, medical_folder_dataset, common

from utils import success, error, validate_json, validate_request_data, response
from schemas import ValidateMedicalFolderReferenceCSV, \
    ValidateMedicalFolderRoot, \
    ValidateMedicalFolderAddRequest, \
    PreviewDatasetRequest

from fedbiomed.common.data import MedicalFolderController, MapperBlock, MedicalFolderLoadingBlocks
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.common.exceptions import FedbiomedError
dataset_manager = DatasetManager()

# Medical Folder Controller
mf_controller = MedicalFolderController()

# Path to write and read the datafiles
DATA_PATH_RW = app.config['DATA_PATH_RW']

# Database table (datasets table of TinyDB) and query object
table = database.db().table_datasets()
query = database.query()


@api.route('/datasets/medical-folder-dataset/validate-reference-column', methods=['POST'])
@validate_request_data(schema=ValidateMedicalFolderReferenceCSV)
@middleware(middlewares=[medical_folder_dataset.read_medical_folder_reference,
                         medical_folder_dataset.load_dlp,
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

    data_path_save = os.path.join(app.config['DATA_PATH_SAVE'], *req['medical_folder_root'])

    # Create unique id for the dataset
    dataset_id = 'dataset_' + str(uuid.uuid4())

    if req["reference_csv_path"] is None:
        dataset_parameters = {}
    else:
        reference_csv = os.path.join(app.config['DATA_PATH_SAVE'], *req["reference_csv_path"])
        dataset_parameters = {"index_col": req["index_col"],
                              "tabular_file": reference_csv}

    try:
        dataset_manager.add_database(name=req["name"],
                                     data_type="medical-folder",
                                     tags=req['tags'],
                                     description=req['desc'],
                                     path=data_path_save,
                                     dataset_id=dataset_id,
                                     dataset_parameters=dataset_parameters,
                                     data_loading_plan=g.dlp)
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


@api.route('/datasets/medical-folder-dataset/preview', methods=['POST'])
@validate_request_data(schema=PreviewDatasetRequest)
@cached(key="dataset_id", prefix="medical_folder_dataset-preview", timeout=600)
def medical_folder_preview():
    """Gets preview of MedicalFolder dataset by providing a table of subject and available modalities"""

    # Request object as JSON
    req = request.json

    dataset = table.get(query.dataset_id == req['dataset_id'])

    # Extract data path where the files are saved in the local GUI repository
    rexp = re.match('^' + app.config['DATA_PATH_SAVE'], dataset['path'])
    data_path = dataset['path'].replace(rexp.group(0), app.config['DATA_PATH_RW'])
    mf_controller.root = data_path

    if "index_col" in dataset["dataset_parameters"]:
        # Extract data path where the files are saved in the local GUI repository
        rexp = re.match('^' + app.config['DATA_PATH_SAVE'], dataset['path'])
        reference_path = dataset["dataset_parameters"]["tabular_file"].replace(rexp.group(0),
                                                                               app.config['DATA_PATH_RW'])

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


@api.route('/datasets/medical-folder-dataset/create-modalities-to-folders-loading-block', methods=['POST'])
def create_modalities_to_folders_pipeline():
    req = request.json
    dp = MapperBlock()
    dp.map = req['mapping']
    key = MedicalFolderLoadingBlocks.MODALITIES_TO_FOLDERS
    dataset_manager.save_data_loading_block(dp)
    return response(data={'serial_id': dp.get_serialization_id(),
                          'module': key.__module__,
                          'qualname': key.__class__.__qualname__}), 200
