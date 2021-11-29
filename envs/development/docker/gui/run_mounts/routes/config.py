import os
import configparser
from posixpath import join
from . import api 
from app import app, db_prefix
from flask import request, jsonify
from utils import get_node_id

def return_error(msg:str):

    return jsonify(
        {
            'success' : False,
            'message' : msg
        }
    )

def return_success(msg:str):

    return jsonify(
        {
            'success' : True,
            'message' : msg
        }
    )

@api.route('/config/change-node-config', methods = ['POST'])
def change_node_config():

    """ Change config file that is going to be used 
        for node GUI. This action changes all database
        queries. It is just development porposes. In productoin
        it is assumed that there will be only one node which only
        has one config file. 
    """

    req = request.json 
    if req['config-file']:
        fullpath = os.path.join(app.config['NODE_FEDBIOMED_ROOT'], 'etc', req['config-file'])
        if os.path.isfile(fullpath):
            node_id = get_node_id(fullpath)
            app.config.update(
                NODE_CONFIG_FILE = req['config-file'],
                NODE_CONFIG_FILE_PATH = fullpath,
                NODE_ID       = node_id, 
                NODE_DB_PATH  = os.path.join(app.config['NODE_FEDBIOMED_ROOT'], 'var', db_prefix + node_id + '.json') 
            )

            return return_success('Node configuration has succesfully changed')
        else:
           return return_error('Config file is does not exist. Please make sure \
               you type your config file correctly')                 
    else:
        return return_error('Missing config-file parameter')

@api.route('/config/node-id', methods = ['GET'])
def node_id():

    """ API enpoint to get node id which GUI will be working for """
    result = { 
            'success' : True,
            'node_id' : app.config['NODE_ID']
            }   

    return jsonify(result)


