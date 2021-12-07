import os
from typing import List 

from flask import jsonify, request
from app import app
from schemas import ListDataFolder


from . import api 
from utils import success, error, validate_json, validate_request_data
from db import database

@api.route('/repository/list' , methods=['POST'])
@validate_json
@validate_request_data(schema = ListDataFolder)
def list_data_path():

    """ API endpoint to list folders in the data path of the node. 
        It only allows the list files in the basis of data path.

    Request: 

        path (list): List that includes folders in hierarchical order. 

    """ 

    input = request.json
    req_path = input['path']
    
    # Full path by including the base DATA_PATH
    dpath = os.path.join(app.config["DATA_PATH"], *req_path) 

    # Check if the path is exist or it is a directory
    if os.path.exists(dpath) and os.path.isdir(dpath):
        base = '/' if len(req_path) == 0 else os.path.join(*req_path)
        pathfiles = os.listdir(dpath)
        res = {
            'level': len(req_path),
            'base' : base,
            'files' : []
        }

        table = database.db().table('_default')
        query = database.query()

        for file in pathfiles:
            fullpath = os.path.join(dpath, file)

            # Get dataset registered with full path
            dataset = table.get(query.path == fullpath)

            # This is the path that will be displayed on the GUI 
            # It is create as list to be able use it with `os.path.join` 
            exact_path = [*req_path , file]
            
            if os.path.isdir(fullpath):
                res['files'].append({"type" : 'dir' , "name": file, "path": exact_path, 'registered' : dataset})
            elif os.path.isfile(fullpath):
                res['files'].append({"type" : 'file' , "name": file, "path": exact_path, 'registered' : dataset})
        
        return jsonify(res), 200

    else:

        return error(f'Reqeusted path does not exist or it is not a directory. {req_path}')


