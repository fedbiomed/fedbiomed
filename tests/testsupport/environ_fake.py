"""
this provides a fake environment for the test

it is a replacement for fedbiomed.common.environ
specifically for the researcher

this module should be imported by

mock_common_environ.py

"""
import os
from posix import listdir

from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.constants import ComponentType
import inspect
import shutil

class Environ(metaclass = SingletonMeta):

    def __init__(self, component = None):

        print("Using fake environ:", component)

        self.envdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                ), '..' , '..' , 'envs'
            )
        self._values={}

        self._values['ROOT_DIR']                = "/tmp"
        self._values['CONFIG_DIR']              = "/tmp/etc"
        self._values['VAR_DIR']                 = "/tmp/var"
        self._values['CACHE_DIR']               = "/tmp/var/cache"
        self._values['TMP_DIR']                 = "/tmp/var/tmp"
        self._values['MQTT_BROKER']             = "localhost"
        self._values['MQTT_BROKER_PORT']        = 1883
        self._values['UPLOADS_URL']             = "http://localhost:8888/upload/"
        self._values['TIMEOUT']                 = 10
        self._values['DEFAULT_MODELS_DIR']      = '/tmp/default_models'

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

        #if component == ComponentType.NODE:
        if True:
            # values specific to node
            self._values['MESSAGES_QUEUE_DIR'] = "/tmp/var/queue_messages_XXX"
            self._values['NODE_ID']          = "mock_node_XXX"
            self._values['DB_PATH']            = '/tmp/var/db_node_mock_node_XXX.json'
            
            
            self._values['ALLOW_DEFAULT_MODELS'] = True
            self._values['MODEL_APPROVAL'] = True
            self._values['HASHING_ALGORITHM'] = 'SHA256'

        #if component == ComponentType.RESEARCHER:
        if True:
            # values specific to researcher
            self._values['MESSAGES_QUEUE_DIR']      = "/tmp/var/queue_messages"
            self._values['RESEARCHER_ID']           = "mock_researcher_XXX"
            self._values['BREAKPOINTS_DIR']         = '/tmp/var/breakpoints'
            self._values['TENSORBOARD_RESULTS_DIR'] = "/tmp/runs"

            os.makedirs(self._values['BREAKPOINTS_DIR']        , exist_ok=True)
            os.makedirs(self._values['TENSORBOARD_RESULTS_DIR'], exist_ok=True)


    def values(self):
        return self._values
