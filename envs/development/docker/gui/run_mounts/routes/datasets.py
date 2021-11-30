from flask import jsonify, request
from . import api 

from db import database
from .helper import success, error, validate_json, validate_request_data
from functools import wraps
from .schemas import AddTabularData

@api.route('/datasets/list' , methods=['POST'])
def list():

    """
        List all datasets saved into database
    """

    table = database.db.table('_default')
    datasets = table.all()
    return jsonify(datasets)

@api.route('/datasets/remove' , methods=['DELETE'])
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

    return 'NONE'




@api.route('/datasets/add-image-dataset', methods=['POST'])
def add_image_dataset():

    """ API endpoint to add image dataset 

    Request.Json: 

        name (str): Name for the dataset
        tags (array): Tags for the dataset
        path (str): Datapath where images are saved

    """