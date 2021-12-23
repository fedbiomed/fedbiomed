import os
import uuid
import re
from flask import jsonify, request
from app import app
from db import database

from . import api
from utils import success, error, validate_json, validate_request_data, response
from schemas import AddDataSetRequest, \
    RemoveDatasetRequest, \
    UpdateDatasetRequest, \
    PreviewDatasetRequest, \
    AddDefaultDatasetRequest, \
    ListDatasetRequest

from fedbiomed.node.data_manager import DataManager

# Initialize Fed-BioMed DataManager
datamanager = DataManager()


@api.route('/datasets/list', methods=['POST'])
@validate_request_data(schema=ListDatasetRequest)
def list_datasets():
    """
    List Datasets saved into Node DB

    Request.GET {None}:
        - No request data

    Response {application/json}:
        400:
            success   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDBt
        200:
            success: Boolean value indicates that the request is success
            result: List of dataset objects
            endpoint: API endpoint
            message: The message for response
    """
    req = request.json
    search = req.get('search', None)
    table = database.db().table('_default')
    query = database.query()

    table.clear_cache()
    if search is not None and search != "":
        res = table.search(query.name.search(search + '+') | query.description.search(search + '+'))
    else:
        try:
            res = table.all()
            database.close()
        except Exception as e:
            return error(str(e)), 400

    return response(res), 200


@api.route('/datasets/remove', methods=['POST'])
@validate_request_data(schema=RemoveDatasetRequest)
def remove_dataset():
    """ API endpoint to remove single dataset from database.
    This method removed dataset from database not from file system.

    Request {application/json}:
        dataset_id (str): Id of the dataset which will be removed

    Response {application/json}:
        400:
            success   : Boolean error status (False)
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        200:
            success : Boolean value indicates that the request is success
            result  : null
            message : The message for response
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


@api.route('/datasets/add', methods=['POST'])
@validate_request_data(schema=AddDataSetRequest)
def add_dataset():
    """ API endpoint to add single dataset to the database. Currently it
        uses some methods of datamanager.

    Request {application/json}:

        name (str): Name for the dataset
        tags (array): Tags for the dataset
        path (array): Data path where dataset is saved
        desc (string): Description for dataset
        type (string): Type of the dataset, CSV or Images

    Response {application/json}:
        400:
            error   : Boolean error status
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB
        200:
            success : Boolean value indicates that the request is success
            result  : Dataset object (TindyDB doc).
            message : The message for response

    """
    table = database.db().table('_default')
    query = database.query()

    data_path_rw = app.config['DATA_PATH_RW']
    req = request.json

    # Data path that the files will be read
    data_path = os.path.join(data_path_rw, *req['path'])

    # Data path that will be saved in the DB
    data_path_save = os.path.join(app.config['DATA_PATH_SAVE'], *req['path'])

    # Get image dataset information from datamanager
    if req['type'] == 'images':
        try:
            shape = datamanager.load_images_dataset(data_path)
            types = []
        except Exception as e:
            return error(str(e)), 400
    # Get csv dataset information from datamanager
    elif req['type'] == 'csv':
        try:
            data = datamanager.load_csv_dataset(data_path)
            shape = data.shape
            types = datamanager.get_csv_data_types(data)
        except Exception as e:
            return error(str(e)), 400
    else:
        return error(f'Unknown dataset type "{req["type"]}"'), 400

    # Create unique id for the dataset
    dataset_id = 'dataset_' + str(uuid.uuid4())

    try:
        table.insert({
            "name": req['name'],
            "path": data_path_save,
            "data_type": req['type'],
            "dtypes": types,
            "shape": shape,
            "tags": req['tags'],
            "description": req['desc'],
            "dataset_id": dataset_id
        })
    except Exception as e:
        return error(str(e)), 400

    # Get saved dataset document
    res = table.get(query.dataset_id == dataset_id)

    return response(res), 200


@api.route('/datasets/update', methods=['POST'])
@validate_request_data(schema=UpdateDatasetRequest)
def update_dataset():
    """API endpoint for updating dataset

    ---
    Request {application/json}:
            name    : name for the dataset minLength:4 and MaxLength:124`
            tags    : new tags for the data sets
            desc    : String, description about dataset

    Response {application/json}:
        400:
            success : False, Boolean success status always, False
            result  : null
            message : Message about error. Can be validation error or
                      error from TinyDB

        200:
            success : Boolean value indicates that the request is success
            result  : Default dataset json object
            message : The message for response
    """
    req = request.json
    table = database.db().table('_default')
    query = database.query()

    table.update({"tags": req["tags"],
                  "description": req["desc"],
                  "name": req["name"]},
                 query.dataset_id == req['dataset_id'])
    res = table.get(query.dataset_id == req['dataset_id'])

    return response(res), 200


@api.route('/datasets/preview', methods=['POST'])
@validate_request_data(schema=PreviewDatasetRequest)
def get_preview_dataset():
    """API endpoint for getting preview information for dataset
    ----
    Request {application/json}:
            dataset_id (str): ID of the dataset that will be
                              previewed

    Response {application/json}:
        400:
            error (bool): Boolean error status
            result (any): null
            message (str): Error message

        200:
            success (bool): Boolean value indicates that the request is success
            result (json): Default dataset json object
            endpoint (str): API endpoint
            message (str): The message for response
    """

    req = request.json
    table = database.db().table('_default')
    query = database.query()
    dataset = table.get(query.dataset_id == req['dataset_id'])

    # Extract data path where the files are saved in the local repository
    rexp = re.match('^' + app.config['DATA_PATH_SAVE'], dataset['path'])
    data_path = dataset['path'].replace(rexp.group(0), app.config['DATA_PATH_RW'])

    if dataset:
        if os.path.isfile(data_path):
            df = datamanager.read_csv(data_path)
            data_preview = df.head().to_dict('split')
            dataset['data_preview'] = data_preview
        elif os.path.isdir(data_path):
            path_root = os.path.normpath(app.config["DATA_PATH_RW"]).split(os.sep)
            path = os.path.normpath(data_path).split(os.sep)
            dataset['data_preview'] = path[len(path_root):len(path)]
        else:
            dataset['data_preview'] = None

        matched = re.match('^' + app.config['NODE_FEDBIOMED_ROOT'], str(dataset['path']))
        if matched:
            dataset['path'] = dataset['path'].replace(app.config['NODE_FEDBIOMED_ROOT'], '$FEDBIOMED_ROOT')

        return response(dataset), 200

    else:
        return error('No data has been found with this id'), 400


@api.route('/datasets/add-default-dataset', methods=['POST'])
@validate_request_data(schema=AddDefaultDatasetRequest)
def add_default_dataset():
    """API endpoint for adding default dataset

    ---

    Request {application/json}:
            name    : name of the default dataset, this parameter is not
                      required since the only default dataset is MNIST. Default value
                      is `mnist` that is generated by `AddDefaultDatasetRequest`

    Response {application/json}:
        400:
            error   : Boolean error status
            result  : null
            message : Message about error. For this API it comes from
                     `DataManager` class of Fed-BioMed.

        200:
            success : Boolean value indicates that the request is success
            result  : Default dataset json object
            endpoint: API endpoint
            message : The message for response

    """
    req = request.json
    table = database.db().table('_default')
    query = database.query()
    dataset = table.get(query.tags == req['tags'])

    if dataset:
        return error(f'An default dataset has been already deployed with {req["tags"]}'), 400


    if 'path' in req:
        # Data path that the files will be read
        path = os.path.join(app.config['DATA_PATH_RW'], *req['path'])
        # This is the path will be writen in DB
        data_path = os.path.join(app.config['DATA_PATH_SAVE'], *req['path'])
    else:
        default_dir = os.path.join(app.config["DATA_PATH_RW"], 'defaults')
        path = os.path.join(default_dir, 'mnist')
        if not os.path.exists(default_dir):
            os.mkdir(default_dir)
        if not os.path.exists(os.path.join(default_dir, 'mnist')):
            os.mkdir(path)

        # This is the path will be writen in DB
        data_path = os.path.join(app.config['DATA_PATH_SAVE'], 'defaults', 'mnist')

    try:
        shape = datamanager.load_default_database(name="MNIST",
                                                  path=path,
                                                  as_dataset=False)
    except Exception as e:
        return error(str(e)), 400

    # Create database connection
    table = database.db().table('_default')
    query = database.query()

    # Create unique id for the dataset
    dataset_id = 'dataset_' + str(uuid.uuid4())

    try:
        table.insert({
            "name": req['name'],
            "path": data_path,
            "data_type": 'default',
            "dtypes": [],
            "shape": shape,
            "tags": req['tags'],
            "description": req['desc'],
            "dataset_id": dataset_id})

    except Exception as e:
        return error(str(e)), 400

    res = table.get(query.dataset_id == dataset_id)

    return response(res), 200
