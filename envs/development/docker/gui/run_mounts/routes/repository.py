import os
from typing import List

from flask import jsonify, request
from app import app
from schemas import ListDataFolder

from . import api
from utils import success, error, validate_json, validate_request_data
from db import database


@api.route('/repository/list', methods=['POST'])
@validate_request_data(schema=ListDataFolder)
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
        base = os.sep if len(req_path) == 0 else os.path.join(*req_path)
        files = os.listdir(dpath)

        res = {
            'level': len(req_path),
            'base': base,
            'files': [],
            'number': len(files),
            'displays': len(files) if len(files) <= 100 else 100
        }

        files = files if len(files) <= 100 else files[0:100]

        table = database.db().table('_default')
        query = database.query()
        table.clear_cache()

        for file in files:
            fullpath = os.path.join(dpath, file)
            # Get dataset registered with full path
            dataset = table.get(query.path == fullpath)

            # Folder that includes any data file
            includes = []
            if not dataset and os.path.isdir(fullpath):
                includes = table.search(query.path.matches('^' + os.path.join(fullpath, '')))

            # This is the path that will be displayed on the GUI 
            # It is created as list to be able to use it with `os.path.join`
            exact_path = [*req_path, file]
            extension = os.path.splitext(fullpath)[1]

            path_type = 'file' if os.path.isfile(fullpath) else 'dir'

            res['files'].append({"type": path_type,
                                 "name": file,
                                 "path": exact_path,
                                 "extension": extension,
                                 'registered': dataset,
                                 'includes': includes})

        return jsonify(res), 200

    else:

        return error(f'Requested path does not exist or it is not a directory. {req_path}')
