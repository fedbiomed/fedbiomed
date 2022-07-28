import re

from app import app
from utils import response

from fedbiomed.node.environ import environ
from flask_jwt_extended import jwt_required, verify_jwt_in_request
from . import api


@api.route('/config/node-id', methods=['GET'])
def node_id():
    """ API enpoint to get node id which GUI will be working for

        Request GET {any}:
            - No request data

        Response {application/json}:
            200:
                success: Boolean value indicates that the request is success
                result: Object containing node_id
                message: The message for response
    """

    result = {
        'node_id': app.config['NODE_ID']
    }
    print('verif', verify_jwt_in_request())

    return response(result), 200


@api.route('/config/node-environ', methods=['GET'])
def fedbiomed_environ():
    """ Endpoint that return current configuration for node

    Request {application/json}:
        - No value

    Response {application/json}:
        200:
            success: Boolean value indicates that the request is success
            result: Object containing configuration values
            message: The message for response
    """
    res = {}
    confs = ['NODE_ID', 'DB_PATH', 'ROOT_DIR',
             'CONFIG_DIR', 'DEFAULT_MODELS_DIR', 'MESSAGES_QUEUE_DIR',
             'MQTT_BROKER', 'MQTT_BROKER_PORT', 'UPLOADS_URL',
             'MODEL_APPROVAL', 'ALLOW_DEFAULT_MODELS', 'HASHING_ALGORITHM']

    for key in confs:
        try:
            res[key] = environ[key]
            matched = re.match('^' + app.config['NODE_FEDBIOMED_ROOT'], str(environ[key]))
            if matched and key is not 'ROOT_DIR':
                res[key] = res[key].replace(app.config['NODE_FEDBIOMED_ROOT'], '$FEDBIOMED_DIR')
        except Exception as e:
            print(f'ERROR: An error occurred while calling /node-environ endpoint - {e} \n')
            pass

    return response(res), 200

# TODO: Should be used when it is required to manage multiple nodes
# from single GUI. Currently when the config is changed some of the 
# Fed-BioMed APIs still use previous node config e.g. DatasetManager.
# @api.route('/config/change-node-config', methods=['POST'])
# def change_node_config():
#     """ Change config file that is going to be used
#         for node GUI. This action changes all database
#         queries. It is just development porposes. In productoin
#         it is assumed that there will be only one node which only
#         has one config file.
#     """
#
#     req = request.json
#     if req['config-file']:
#         fullpath = os.path.join(app.app.config['NODE_FEDBIOMED_ROOT'], 'etc', req['config-file'])
#         if os.path.isfile(fullpath):
#             node_id = get_node_id(fullpath)
#
#             # Reload environ after updating CONFIG_FILE environment variable
#             os.environ['CONFIG_FILE'] = fullpath
#             reload(fedbiomed.common.environ)
#             reload(fedbiomed.node.environ)
#             reload(fedbiomed.node.dataset_manager)
#
#             app.config.update(
#                 NODE_CONFIG_FILE=req['config-file'],
#                 NODE_CONFIG_FILE_PATH=fullpath,
#                 NODE_ID=node_id,
#                 NODE_DB_PATH=os.path.join(app.config['NODE_FEDBIOMED_ROOT'], 'var', db_prefix + node_id + '.json')
#             )
#
#             return success('Node configuration has succesfully changed')
#         else:
#             return error('Config file is does not exist. Please make sure' + \
#                          ' you type your config file correctly')
#     else:
#         return error('Missing config-file parameter')
