import os
import uuid

from flask import jsonify, request
from app import app
from db import database

from . import api
from utils import success, error, validate_json, validate_request_data, response
from schemas import AddDataSetRequest, \
                    RemoveDatasetRequest, \
                    UpdateDatasetRequest, \
                    PreviewDatasetRequest

from fedbiomed.node.data_manager import DataManager

# Initialize Fed-BioMed DataManager
datamanager = DataManager()


@api.route('/datasets/list', methods=['POST'])
def list():

    """
        List all datasets saved into database
    """

    table = database.db().table('_default')
    table.clear_cache()
    res = table.all()
    database.close()

    return response(res, '/api/datasets/add-csv'), 200


@api.route('/datasets/remove', methods=['POST'])
@validate_json
@validate_request_data(schema=RemoveDatasetRequest)
def remove():

    """ API endpoint to remove single dataset from database. 
    This method removed dataset from database not from file system.

    Request.Json:

        dataset_id (str): Id of the dataset which will be removed
    """
    req = request.json

    if req['dataset_id']:

        table = database.db().table('_default')
        query = database.query()
        dataset = table.get(query.dataset_id == req['dataset_id'])

        if dataset:
            table.remove(doc_ids=[dataset.doc_id])
            database.close()
            return success('Dataset has been removed successfully'), 200

        else:
            database.close()
            return error('Can not find specified dataset in the database'), 400
    else:
        return error('Missing `dataset_id` attribute.'), 400


@api.route('/datasets/add-csv', methods=['POST'])
@validate_json
@validate_request_data(schema=AddDataSetRequest)
def add_csv_dataset():

    """ API endpoint to add single dataset to the database. Currently it 
        uses some of the methods of datamanager.

    Request.Json:

        name (str): Name for the dataset
        tags (array): Tags for the dataset
        path (array): Datapath where dataset is saved
        desc (string): Description for dataset
        type (string): Type of the dataset, CSV or Images

    """
    base_data_path = app.config['DATA_PATH']
    input = request.json

    data_path = os.path.join(base_data_path, *input['path'])

    try:
        dataset_id = datamanager.add_database(
                    name=input['name'],
                    data_type=input['type'],
                    tags=input['tags'],
                    description=input['desc'],
                    path=data_path,
        )
    except Exception as e:
        return error(str(e))

    table = database.db().table('_default')
    query = database.query()
    res = table.get(query.dataset_id == dataset_id)

    return response(res, '/api/datasets/add-csv'), 200


@api.route('/datasets/update', methods=['POST'])
@validate_json
@validate_request_data(schema=UpdateDatasetRequest)
def update_dataset():
    input = request.json
    table = database.db().table('_default')
    query = database.query()

    table.update({"tags": input["tags"],
                  "description": input["desc"],
                  "name": input["name"]},
                 query.dataset_id == input['dataset_id'])

    return None, 200


@api.route('/datasets/preview', methods=['POST'])
@validate_json
@validate_request_data(schema=PreviewDatasetRequest)
def get_preview_dataset():

    input = request.json
    table = database.db().table('_default')
    query = database.query()
    dataset = table.get(query.dataset_id == input['dataset_id'])

    if dataset:
        if os.path.isfile(dataset['path']):
            df = datamanager.read_csv(dataset['path'])
            data_preview = df.head().to_dict('split')
            dataset['data_preview'] = data_preview
        else:
            dataset['data_preview'] = None
        return response(dataset, '/api/datasets/preview'), 200

    else:
        return error('No data has been found with this id'), 400
