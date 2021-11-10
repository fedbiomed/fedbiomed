import configparser
import os
import sys
import uuid

from fedbiomed.common.logger         import logger
from fedbiomed.common.singleton      import SingletonMeta
from fedbiomed.common.component_type import ComponentType


"""
Descriptions of global/environment variables 

Resarcher Global Variables: 
    RESEARCHER_ID           : id of the researcher
    ID                      : equals to researcher id
    TENSORBOARD_RESULTS_DIR : path for writing tensorboard log files
    BREAKPOINTS_DIR         : folder for saving breakpoints 
    MESSAGES_QUEUE_DIR      : Path for writing queue files

Nodes Global Variables:
    NODE_ID                 : id of the node
    ID                      : equals to node id
    MESSAGES_QUEUE_DIR      : Path for queues
    DB_PATH                 : TinyDB database path  

Common Global Variables: 
    COMPONENT_TYPE          : Node or Researcher
    CONFIG_DIR              : Configuration file path
    VAR_DIR                 : Var directory of Fed-Biomed
    CACHE_DIR               : Cache directory of Fed-BioMed
    TMP_DIR                 : Temporary directory 
    MQTT_BROKER             : MQTT broker IP address
    MQTT_BROKER_PORT        : MQTT broker port   
    UPLOADS_URL             : Upload URL for file repository
"""


class Environ(metaclass = SingletonMeta):
    """
    this (singleton) class contains all variables for researcher or node
    """

    def __init__(self, component = None):
        """
        class constructor

        input: type of the component either ComponentType.NODE or ComponentType.RESEARCHER
        """
        # dict with contains all configuration values
        self._values = {}

        if component == ComponentType.NODE or component == ComponentType.RESEARCHER :
            self._values['COMPONENT_TYPE'] = component
        else:
            logger.critical("Environ() parameter should be of ComponentType")
            sys.exit(-1)

        # common values for all components
        self._init_common()

        # must be read before specific configuration values
        # which may beed the ID of the node to be specified
        self._parse_config_file()

        # specific configuration values
        if component == ComponentType.RESEARCHER:
            logger.setLevel("DEBUG")
            self._init_researcher()

        if component == ComponentType.NODE:
            self._init_node()



    def _init_common(self):
        """
        commun configuration values for researcher and node
        """

        # locate the top dir from the file location
        ROOT_DIR = os.path.abspath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                '../..'
            )
        )
        self._values['ROOT_DIR'] = ROOT_DIR

        # main directories
        self._values['CONFIG_DIR'] = os.path.join(ROOT_DIR, 'etc')
        VAR_DIR = os.path.join(ROOT_DIR, 'var')
        self._values['VAR_DIR']    = VAR_DIR
        self._values['CACHE_DIR']  = os.path.join(VAR_DIR, 'cache')
        self._values['TMP_DIR']    = os.path.join(VAR_DIR, 'tmp')

        for _key in 'CONFIG_DIR', 'VAR_DIR', 'CACHE_DIR', 'TMP_DIR':
            dir = self._values[_key]
            if not os.path.isdir(dir):
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    logger.error("path exists but is not a directory " + dir)

        pass


    def _parse_config_file(self):
        """
        read the .ini file corresponding to the component
        create the file with default values if it does not exists

        complete/modify the environment with values coming from the
        OS environment variables
        """

        # get config file location from environment
        # or use a predefined value
        if os.getenv('CONFIG_FILE') :
            CONFIG_FILE = os.getenv('CONFIG_FILE')
            if not os.path.isabs(CONFIG_FILE):
                CONFIG_FILE = os.path.join(self._values['CONFIG_DIR'],
                                           os.getenv('CONFIG_FILE'))
        else:
            if self._values['COMPONENT_TYPE'] == ComponentType.RESEARCHER:
                CONFIG_FILE = os.path.join(self._values['CONFIG_DIR'],
                                           'config_researcher.ini')
            else:
                CONFIG_FILE = os.path.join(self._values['CONFIG_DIR'],
                                           'config_node.ini')


        # parser for the .ini file
        cfg = configparser.ConfigParser()

        if os.path.isfile(CONFIG_FILE):
            # get values from .ini file
            cfg.read(CONFIG_FILE)
        else:
            self._write_config_file(cfg, CONFIG_FILE)
        
    
        # store the CONFIG_FILE in environ (may help to debug)
        self._values['CONFIG_FILE'] = CONFIG_FILE

        #
        # values from the config file may also be overloaded with
        # OS environment variables
        #

        # component ID
        if self._values['COMPONENT_TYPE'] == ComponentType.RESEARCHER:
            # we may remove RESEARCHER_ID in the future (to simplify the code)
            # and use ID instead
            self._values['RESEARCHER_ID'] = os.getenv('RESEARCHER_ID',
                                                      cfg.get('default',
                                                              'researcher_id'))
            self._values['ID'] = self._values['RESEARCHER_ID']
        else:
            # we may remove NODE_ID in the future (to simplify the code)
            # and use ID instead
            self._values['NODE_ID']  = os.getenv('NODE_ID',
                                                   cfg.get('default',
                                                           'node_id'))
            self._values['ID'] = self._values['NODE_ID']

        # broker location
        self._values['MQTT_BROKER'] = os.getenv('MQTT_BROKER',
                                                cfg.get('mqtt',
                                                        'broker_ip'))
        self._values['MQTT_BROKER_PORT']  = int(os.getenv('MQTT_BROKER_PORT',
                                                          cfg.get(
                                                              'mqtt',
                                                              'port')))

        # repository location
        UPLOADS_URL = cfg.get('default', 'uploads_url')
        uploads_ip = os.getenv('UPLOADS_IP')
        if uploads_ip:
            UPLOADS_URL = "http://" + uploads_ip + ":8844/upload/"
            UPLOADS_URL = os.getenv('UPLOADS_URL', UPLOADS_URL)

        # trailing slash is needed for repo url
        if not UPLOADS_URL.endswith('/'):
            UPLOADS_URL += '/'

        self._values['UPLOADS_URL'] = UPLOADS_URL
        self._values['TIMEOUT']     = 5

        pass

    def _init_researcher(self):
        """
        specific configuration values for researcher
        """
        ROOT_DIR = self._values['ROOT_DIR']
        VAR_DIR = self._values['VAR_DIR']

        # more directories
        self._values['TENSORBOARD_RESULTS_DIR'] = os.path.join(ROOT_DIR, 'runs')
        self._values['BREAKPOINTS_DIR'] = os.path.join(VAR_DIR, "breakpoints")

        for _key in 'TENSORBOARD_RESULTS_DIR', 'BREAKPOINTS_DIR':
            dir = self._values[_key]
            if not os.path.isdir(dir):
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    logger.error("path exists but is not a directory " + dir)

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(
            VAR_DIR,
            'queue_messages')
        pass

    def _init_node(self):
        """
        specific configuration values for node
        """

        VAR_DIR = self._values['VAR_DIR']
        NODE_ID = self._values['NODE_ID']

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(VAR_DIR,
                                                          f'queue_manager_{NODE_ID}')
        self._values['DB_PATH'] = os.path.join(VAR_DIR,
                                               f'db_{NODE_ID}.json')



        # ========= PATCH MNIST Bug torchvision 0.9.0 ===================
        # https://github.com/pytorch/vision/issues/1938

        # imported only for the node component
        from six.moves import urllib

        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-agent', 'Python-urllib/3.7'),
            ('Accept',
             'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
            ('Accept-Language', 'en-US,en;q=0.9'),
            ('Accept-Encoding', 'gzip, deflate, br')
        ]
        urllib.request.install_opener(opener)

        pass

    def _write_config_file(self, cfg, config_file):

        """This method writes new config file"""

        # use default values from current OS environment variables

        # repository location
        uploads_url = "http://localhost:8844/upload/"
        uploads_ip = os.getenv('UPLOADS_IP')
        if uploads_ip:
            uploads_url = "http://" + uploads_ip + ":8844/upload/"
        uploads_url = os.getenv('UPLOADS_URL', uploads_url)

        # Write based on component type
        if self._values['COMPONENT_TYPE'] == ComponentType.RESEARCHER:
            # we may remove researcher_id in the future (to simplify the code)
            # and use id instead
            researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))
            cfg['default'] = {
                'researcher_id': researcher_id,
                'uploads_url': uploads_url
            }
        else:
            # TODO: We may remove node_id in the future (to simplify the code)
            node_id = os.getenv('NODE_ID', 'node_' + str(uuid.uuid4()))

            cfg['default'] = {
                'node_id': node_id,
                'uploads_url': uploads_url
            }

        # message broker
        mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        mqtt_broker_port = int(os.getenv('MQTT_BROKER_PORT', 1883))

        cfg['mqtt'] = {
            'broker_ip': mqtt_broker,
            'port': mqtt_broker_port,
            'keep_alive': 60
        }

        # write the config for future relaunch of the same component
        # (only if the file does not exists)
        try:
            with open(config_file, 'w') as f:
                cfg.write(f)
        except:
            logger.error("cannot save config file: " + config_file)         

    def values(self):
        return self._values

    def print_component_type(self):
        """
        salutation function (for debug purpose mainly)
        """
        print("I am a:", self._values['COMPONENT_TYPE'])
