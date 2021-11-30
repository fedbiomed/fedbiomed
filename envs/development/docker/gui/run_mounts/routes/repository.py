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

    """ API endpoint to remove single dataset from database. 
    This method removed dataset from database not from file system. 

    Request.Json: 
    
        Folder/Files (list): Folders and files in the repository
    """

    input = request.json
    req_path = input['path']
    
    print(input['path'])
    dpath = os.path.join(app.config["DATA_PATH"], *req_path) 

    pathfiles = os.listdir(dpath)
    res = []

    for file in pathfiles:
        fullpath = os.path.join(dpath, file)
        path_to_display = os.path.join(*req_path, file)
        if os.path.isdir(fullpath):
            res.append({"type" : 'dir' , "name": file, "path": path_to_display})
        elif os.path.isfile(fullpath):
            res.append({"type" : 'file' , "name": file, "path": path_to_display})

    return jsonify(res), 200
