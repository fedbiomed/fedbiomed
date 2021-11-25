from app import app, database
from flask import json, render_template, request, jsonify
import pandas as pd
import csv
import os
import uuid


host_root = os.getenv('DATA_HOST_PATH')


print(host_root)
@app.route("/")
def hello_world():

    table = database.table('_default')
    result = table.all()
    
    return jsonify(result)

@app.route('/upload-csv', methods=['POST'])
def upload_file():

    if request.method == 'POST':
        file = request.files['file']
        tags = request.form.getlist('tags')
        name = request.form.get('name')
        desc = request.form.get('desc')
        type = request.form.get('type')

        result = {
            'success' : 0,
            'tags' : tags,
            'name' : name,
            'desc' : desc,
            'type' : type
        }

        path = os.path.join('/fedbiomed', 'data' , file.filename)
        path_host = os.path.join(host_root , 'data' , file.filename)
        file = file.save(os.path.join('/fedbiomed/data' , file.filename))
        sniffer = csv.Sniffer()

        with open(path, 'r') as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            header = None if not sniffer.has_header(file.read()) else 0

        data = pd.read_csv(path, index_col=None, sep=delimiter, header=header)
        shape = data.shape

        table = database.table('_default')
        dataset_id = 'dataset_' + str(uuid.uuid4())
        types = [str(t) for t in data.dtypes]

        table.insert(
            dict(name=name, data_type=type, tags=tags,
            description=desc, shape=shape,
            path=path_host, dataset_id=dataset_id, dtypes=types)
        )
        
        return jsonify(result)
    else:

        return 0

@app.route('/list-datasets', methods=['GET'])
def list_datasets():

    table = database.table('_default')
    result = table.all()
    return jsonify(result)