from flask_login import login_required
from app import app
from flask import request, jsonify
import pandas as pd
import csv
import os
import uuid

from db import node_database


# TODO: Support upload function for user
@app.route('/upload-csv', methods=['POST'])
@login_required
def upload_file():

    if request.method == 'POST':
        file = request.files['file']
        tags = request.form.getlist('tags')
        name = request.form.get('name')
        desc = request.form.get('desc')
        type = request.form.get('type')

        result = {
            'success': 0,
            'tags': tags,
            'name': name,
            'desc': desc,
            'type': type
        }

        path = os.path.join('/fedbiomed', 'data' , file.filename)
        path_host = os.path.join(app.config['NODE_FEDBIOMED_ROOT'] , 'data' , file.filename)
        file.save(os.path.join('/fedbiomed/data' , file.filename))
        sniffer = csv.Sniffer()

        with open(path, 'r') as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            header = None if not sniffer.has_header(file.read()) else 0

        data = pd.read_csv(path, index_col=None, sep=delimiter, header=header)
        shape = data.shape

        table = node_database.table('_default')
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
