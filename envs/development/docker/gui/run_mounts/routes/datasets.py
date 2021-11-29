from . import api 
from db import database
from flask import jsonify



@api.route('/datasets/list' , methods=['POST'])
def list():

    """
        List all datasets saved into database
    """

    table = database.db.table('_default')
    datasets = table.all()
    return jsonify(datasets)

@api.route('/datasets/remove' , methods=['GET'])
def datasets_me():
    
    table = database.db.table('_default')

