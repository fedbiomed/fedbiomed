import os 
import uuid
from flask import jsonify, request
from app import app
from db import database

from . import api 
from .helper import success, error, validate_json, validate_request_data
from .schemas import AddTabularData, RemoveDatasetRequest
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
@validate_request_data(schema=AddTabularData)
def add_csv_dataset():

    """ API endpoint to add single dataset to the database 

    Request.Json: 

        name (str): Name for the dataset
        tags (array): Tags for the dataset
        path (str): Datapath where dataset is saved

    """
    base_data_path = app.config['DATA_PATH']
    input = request.json

    data_path = os.path.join(base_data_path, *input['path'])

  
    table = database.db.table('_default')
    query = database.query

    table.clear_cache()
    
    # Check tags already exist
    taged = table.search(query.tags.all(input['tags']))
    if len(taged) > 0:
        return error('Data tags must be unique')

    try:
        dataset = datamanager.load_csv_dataset(data_path)
        dtypes = datamanager.get_csv_data_types(dataset)

    except Exception as e:
        return error(str(e))


    dataset_id = 'dataset_' + str(uuid.uuid4())
    dataset = dict(
        name          = input['name'],
        data_type     = 'csv',
        tags          = input['tags'],
        description   = input['desc'],
        path          = data_path,
        dataset_id    = dataset_id,
        dtypes        = dtypes,
        shape         = dataset.shape
    )

    table.insert(dataset)
    res = table.get(query.dataset_id == dataset_id)
    
    return jsonify(res)




@api.route('/datasets/add-image-dataset', methods=['POST'])
def add_image_dataset():

    """ API endpoint to add image dataset 

    Request.Json: 

        name (str): Name for the dataset
        tags (array): Tags for the dataset
        path (str): Datapath where images are saved

    """