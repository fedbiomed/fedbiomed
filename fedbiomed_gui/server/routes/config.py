from .api import api
from ..utils import response
from ..config import config

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
    confs = [ ('default', 'id'), ('default', 'db'),
              ('security', 'training_plan_approval'),
              ('security', 'allow_default_training_plans'),
              ('security', 'hashing_algorithm')]

    for section,key in confs:
        try:
            res[key] = config.node_config.get(section, key)
        except Exception as e:
            print(f'ERROR: An error occurred while calling /node-environ endpoint - {e} \n')
            pass

    return response(res), 200

