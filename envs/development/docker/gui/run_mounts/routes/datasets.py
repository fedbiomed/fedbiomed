import os 
import uuid
from flask import jsonify, request
from app import app
from db import database

from . import api 
from utils import success, error, validate_json, validate_request_data
from schemas import AddDataSet, RemoveDatasetRequest
from fedbiomed.node.data_manager import DataManager

# Initialize Fed-BioMed DataManager
datamanager = DataManager()


@api.route('/datasets/list' , methods=['POST'])
def list():

    """
        List all datasets saved into database
    """

    table = database.db.table('_default')
    datasets = table.all()

    return jsonify(datasets)

@api.route('/datasets/remove' , methods=['DELETE'])
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

        table = database.db.table('_default')
        query = database.query
        dataset = table.get(query.dataset_id == req['dataset_id'])
        
        
        if dataset:
            table.remove(doc_ids=[dataset.doc_id])
            result = success('Dataset has been removed successfully')
        else:
            result = error('Can not find specified dataset in the database')

        database.close()

        return result


    else:
        return error('Missing `dataset_id` attribute.')



@api.route('/datasets/add-csv', methods=['POST'])
@validate_json
@validate_request_data(schema=AddDataSet)
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
                    name          = input['name'],
                    data_type     = input['type'],
                    tags          = input['tags'],
                    description   = input['desc'],
                    path          = data_path,
        )
    except Exception as e:
        return error(str(e))

    table = database.db.table('_default')
    query = database.query
    res = table.get(query.dataset_id == dataset_id)
    
    return jsonify(res)

