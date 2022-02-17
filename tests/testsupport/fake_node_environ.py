"""
this provides a fake environment for the test on NODE

must be imported via mock_node_environ
"""
import os
import inspect
import shutil

from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.singleton  import SingletonMeta
from fedbiomed.common.constants  import ComponentType
from fedbiomed.common.logger     import logger

#class Environ(metaclass = SingletonMeta):
class EnvironNode(metaclass = SingletonMeta):

    def __init__(self, component = None):

        print("===== using fake environ:", component)

        if component != ComponentType.NODE :
            raise FedbiomedEnvironError("fake_node_environ: component type must be NODE")

        self.envdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                ), '..' , '..' , 'envs'
            )
        self._values={}

        # TODO: use os.path.join instead of / in path
        # TODO: use os.mktemp instead of /tmp
        self._values['ROOT_DIR']                = "/tmp/_nod_"
        self._values['CONFIG_DIR']              = "/tmp/_nod_/etc"
        self._values['VAR_DIR']                 = "/tmp/_nod_/var"
        self._values['CACHE_DIR']               = "/tmp/_nod_/var/cache"
        self._values['TMP_DIR']                 = "/tmp/_nod_/var/tmp"
        self._values['MQTT_BROKER']             = "localhost"
        self._values['MQTT_BROKER_PORT']        = 1883
        self._values['UPLOADS_URL']             = "http://localhost:8888/upload/"
        self._values['TIMEOUT']                 = 10
        self._values['DEFAULT_MODELS_DIR']      = '/tmp/_nod_/default_models'

        # TODO: create random directory paths like for test_taskqueue.py
        os.makedirs(self._values['ROOT_DIR']               , exist_ok=True)
        os.makedirs(self._values['CONFIG_DIR']             , exist_ok=True)
        os.makedirs(self._values['VAR_DIR']                , exist_ok=True)
        os.makedirs(self._values['CACHE_DIR']              , exist_ok=True)
        os.makedirs(self._values['TMP_DIR']                , exist_ok=True)
        os.makedirs(self._values['DEFAULT_MODELS_DIR']     , exist_ok=True)


        #  Copy default model files to tmp directory for test
        default_models_path = os.path.join(self.envdir, 'development' , 'default_models')
        files = os.listdir(default_models_path)
        for f in files:
            shutil.copy(os.path.join(default_models_path , f) , self._values['DEFAULT_MODELS_DIR'])

        # values specific to node
        self._values['MESSAGES_QUEUE_DIR'] = "/tmp/_nod_/var/queue_messages_XXX"
        self._values['NODE_ID']          = "mock_node_XXX"
        self._values['DB_PATH']            = '/tmp/_nod_/var/db_node_mock_node_XXX.json'


        self._values['ALLOW_DEFAULT_MODELS'] = True
        self._values['MODEL_APPROVAL'] = True
        self._values['HASHING_ALGORITHM'] = 'SHA256'


    def __getitem__(self, key):
        return self._values[key]


    def __setitem__(self, key, value):
        if value is None:
            logger.critical("setting Environ() value to None for key: " + str(key))
            raise FedbiomedEnvironError("setting Environ() value to None for key: " + str(key))

        self._values[key] = value
        return value


environ=EnvironNode(ComponentType.NODE)
