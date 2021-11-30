import os
from typing import List 

from flask import jsonify, request
from db import database
from app import app

from . import api 
from .helper import success, error, validate_json, validate_request_data
from .schemas import ListDataFolder

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

        pathfiles = os.listdir(dpath)
        res = []

        for file in pathfiles:
            fullpath = os.path.join(dpath, file)

            # This is the path that will be displayed on the GUI 
            # It is create as list to be able use it with `os.path.join` 
            path_to_display = req_path.append(file)

            if os.path.isdir(fullpath):
                res.append({"type" : 'dir' , "name": file, "path": path_to_display})
            elif os.path.isfile(fullpath):
                res.append({"type" : 'file' , "name": file, "path": path_to_display})

        return jsonify(res), 200

    else:

        return error(f'Reqeusted path does not exist or it is not a directory. {req_path}')