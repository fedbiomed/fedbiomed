"""
this provides a fake environment for the test on RESEARCHER

must be imported via mock_researcher_environ
"""
import os
import inspect
import shutil
import uuid
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.logger import logger


class ResearcherEnviron:

    def __init__(self):


        self.envdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ), '..', '..', 'envs'
        )
        self._values = {}

        res = f"_res_{uuid.uuid4()}"
        # TODO: use os.path.join instead of / in path
        # TODO: use os.mktemp instead of /tmp
        self._values['ROOT_DIR'] = f"/tmp/{res}"
        self._values['CONFIG_DIR'] = f"/tmp/{res}/etc"
        self._values['CERT_DIR'] = f"/tmp/{res}/etc/certs"
        self._values['VAR_DIR'] = f"/tmp/{res}/var"
        self._values['CACHE_DIR'] = f"/tmp/{res}/var/cache"
        self._values['TMP_DIR'] = f"/tmp/{res}/var/tmp"
        self._values['MQTT_BROKER'] = "localhost"
        self._values['MQTT_BROKER_PORT'] = 1883
        self._values['UPLOADS_URL'] = "http://localhost:8888/upload/"
        self._values['TIMEOUT'] = 10
        self._values['DEFAULT_TRAINING_PLANS_DIR'] = f'/tmp/{res}/default_training_plans'

        # TODO: create random directory paths like for test_taskqueue.py
        os.makedirs(self._values['ROOT_DIR'], exist_ok=True)
        os.makedirs(self._values['CONFIG_DIR'], exist_ok=True)
        os.makedirs(self._values['VAR_DIR'], exist_ok=True)
        os.makedirs(self._values['CACHE_DIR'], exist_ok=True)
        os.makedirs(self._values['TMP_DIR'], exist_ok=True)
        os.makedirs(self._values['DEFAULT_TRAINING_PLANS_DIR'], exist_ok=True)

        #  Copy default model files to tmp directory for test
        default_models_path = os.path.join(self.envdir, 'common', 'default_training_plans')
        files = os.listdir(default_models_path)
        for f in files:
            shutil.copy(os.path.join(default_models_path, f), self._values['DEFAULT_TRAINING_PLANS_DIR'])

        # values specific to researcher
        self._values['MESSAGES_QUEUE_DIR'] = f"/tmp/{res}/var/queue_messages"
        self._values['RESEARCHER_ID'] = f"mock_researcher_{res}_XXX"
        self._values['ID'] = f"mock_researcher_{res}_XXX"
        self._values['DB_PATH'] = f"/tmp/{res}/var/db_researcher_mock_node_XXX.json"
        self._values['EXPERIMENTS_DIR'] = f'/tmp/{res}/var/experiments'
        self._values['TENSORBOARD_RESULTS_DIR'] = f"/tmp/{res}/runs"

        os.makedirs(self._values['EXPERIMENTS_DIR'], exist_ok=True)
        os.makedirs(self._values['TENSORBOARD_RESULTS_DIR'], exist_ok=True)

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        if value is None:
            logger.critical("setting Environ() value to None for key: " + str(key))
            raise FedbiomedEnvironError("setting Environ() value to None for key: " + str(key))

        self._values[key] = value
        return value


class ResearcherRandomEnv(ResearcherEnviron):
    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, key, value):
        self._values[key] = value
        return value


environ = ResearcherEnviron()
