import os
import sys
import configparser
import copy
from importlib import reload
from posixpath import join
from . import api 
from app import app, db_prefix
from flask import request, jsonify
from utils import get_node_id
from .helper import success, error

import fedbiomed.node.environ
import fedbiomed.common.environ
import fedbiomed.node.data_manager
import app

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
        fullpath = os.path.join(app.app.config['NODE_FEDBIOMED_ROOT'], 'etc', req['config-file'])
        if os.path.isfile(fullpath):
            node_id = get_node_id(fullpath)

            # Reload environ after updating CONFIG_FILE environment variable
            os.environ['CONFIG_FILE'] = fullpath
            reload(fedbiomed.common.environ)
            reload(fedbiomed.node.environ)
            reload(fedbiomed.node.data_manager)
            reload(app)

            app.config.update(
                NODE_CONFIG_FILE = req['config-file'],
                NODE_CONFIG_FILE_PATH = fullpath,
                NODE_ID       = node_id, 
                NODE_DB_PATH  = os.path.join(app.config['NODE_FEDBIOMED_ROOT'], 'var', db_prefix + node_id + '.json') 
            )

            return success('Node configuration has succesfully changed')
        else:
           return error('Config file is does not exist. Please make sure' + \
                                    ' you type your config file correctly')                 
    else:
        return error('Missing config-file parameter')

@api.route('/config/node-id', methods = ['GET'])
def node_id():

    """ API enpoint to get node id which GUI will be working for """

    result = { 
            'success' : True,
            'node_id' : app.config['NODE_ID']
            }   

    return jsonify(result)

@api.route('/config/node-environ')
def fedibomed_environ():

    """ Endpoint that return current configuration for node 
    
        Returns: 
            res : Json formated environ object of the node, with status code 
                  200. 
    """

    res = copy.deepcopy(fedbiomed.node.environ.environ)
    pops = ['COMPONENT_TYPE']
    for p in pops:
        res.pop(p)

    return jsonify(res), 200
