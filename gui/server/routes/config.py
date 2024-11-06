import re
from flask_jwt_extended import jwt_required

from fedbiomed.node.environ import environ

from . import api
from ..utils import response

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
        'node_id': config['ID']
    }

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
    confs = ['ID', 'DB_PATH', 'ROOT_DIR',
             'CONFIG_DIR', 'DEFAULT_TRAINING_PLANS_DIR', 'MESSAGES_QUEUE_DIR',
             'TRAINING_PLAN_APPROVAL', 'ALLOW_DEFAULT_TRAINING_PLANS', 'HASHING_ALGORITHM']

    for key in confs:
        try:
            res[key] = environ[key]
            matched = re.match('^' + config['NODE_FEDBIOMED_ROOT'], str(environ[key]))
            if matched and key != 'ROOT_DIR':
                res[key] = res[key].replace(config['NODE_FEDBIOMED_ROOT'], '$FEDBIOMED_DIR')
        except Exception as e:
            print(f'ERROR: An error occurred while calling /node-environ endpoint - {e} \n')
            pass

    return response(res), 200

