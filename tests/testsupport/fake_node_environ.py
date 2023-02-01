"""
this provides a fake environment for the test on NODE

must be imported via mock_node_environ
"""
import os
import inspect
import shutil
import uuid

from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.constants import ComponentType
from fedbiomed.common.logger import logger


class NodeEnviron:

    def __init__(self):

        self.envdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                ), '..', '..', 'envs'
            )
        self._values = {}

        # TODO: use os.path.join instead of / in path
        # TODO: use os.mktemp instead of /tmp
        node = f"_nod_{uuid.uuid4()}"
        self._values['ROOT_DIR'] = f"/tmp/{node}"
        self._values['CONFIG_DIR'] = f"/tmp/{node}/etc"
        self._values['VAR_DIR'] = f"/tmp/{node}/var"
        self._values['CACHE_DIR'] = f"/tmp/{node}/var/cache"
        self._values['TMP_DIR'] = f"/tmp/{node}/var/tmp"
        self._values['MQTT_BROKER'] = "localhost"
        self._values['MQTT_BROKER_PORT'] = 1883
        self._values['UPLOADS_URL'] = "http://localhost:8888/upload/"
        self._values['TIMEOUT'] = 10
        self._values['DEFAULT_TRAINING_PLANS_DIR'] = f"/tmp/{node}/default_training_plans"
        self._values['TRAINING_PLANS_DIR'] = f"/tmp/{node}/registered_training_plans"

        # TODO: create random directory paths like  for test_taskqueue.py
        os.makedirs(self._values['ROOT_DIR'], exist_ok=True)
        os.makedirs(self._values['CONFIG_DIR'], exist_ok=True)
        os.makedirs(self._values['VAR_DIR'], exist_ok=True)
        os.makedirs(self._values['CACHE_DIR'], exist_ok=True)
        os.makedirs(self._values['TMP_DIR'], exist_ok=True)
        os.makedirs(self._values['DEFAULT_TRAINING_PLANS_DIR'], exist_ok=True)
        os.makedirs(self._values['TRAINING_PLANS_DIR'], exist_ok=True)

        #  Copy default model files to tmp directory for test
        default_models_path = os.path.join(self.envdir, 'common', 'default_training_plans')
        files = os.listdir(default_models_path)
        for f in files:
            shutil.copy(os.path.join(default_models_path, f), self._values['DEFAULT_TRAINING_PLANS_DIR'])

        # values specific to node
        self._values['MESSAGES_QUEUE_DIR'] = f"/tmp/{node}/var/queue_messages_XXX"
        self._values['NODE_ID'] = f"mock_node_{node}_XXX"
        self._values['DB_PATH'] = f"/tmp/{node}/var/db_node_mock_node_XXX.json"

        self._values['ALLOW_DEFAULT_TRAINING_PLANS'] = True
        self._values['TRAINING_PLAN_APPROVAL'] = True
        self._values['HASHING_ALGORITHM'] = 'SHA256'

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        if value is None:
            logger.critical("setting Environ() value to None for key: " + str(key))
            raise FedbiomedEnvironError("setting Environ() value to None for key: " + str(key))

        self._values[key] = value
        return value


class NodeRandomEnv(NodeEnviron):
    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, key, value):
        self._values[key] = value
        return value

environ = NodeEnviron()

