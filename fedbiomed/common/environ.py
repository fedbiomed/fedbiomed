# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
All environment/configuration variables are provided by the
**Environ** dictionary.

**Environ** is a singleton class, meaning that only an instance of Environ is available.

Descriptions of global/environment variables

Researcher Global Variables:

- RESEARCHER_ID           : id of the researcher
- ID                      : equals to researcher id
- TENSORBOARD_RESULTS_DIR : path for writing tensorboard log files
- EXPERIMENTS_DIR         : folder for saving experiments
- MESSAGES_QUEUE_DIR      : Path for writing queue files

Nodes Global Variables:

- NODE_ID                           : id of the node
- ID                                : equals to node id
- MESSAGES_QUEUE_DIR                : Path for queues
- DB_PATH                           : TinyDB database path where datasets/training_plans/loading plans are saved
- DEFAULT_TRAINING_PLANS_DIR        : Path of directory for storing default training plans
- TRAINING_PLANS_DIR                 : Path of directory for storing registered training plans
- TRAINING_PLAN_APPROVAL            : True if the node enables training plan approval
- ALLOW_DEFAULT_TRAINING_PLANS      : True if the node enables default training plans for training plan approval

Common Global Variables:

- COMPONENT_TYPE          : Node or Researcher
- CONFIG_DIR              : Configuration file path
- VAR_DIR                 : Var directory of Fed-BioMed
- CACHE_DIR               : Cache directory of Fed-BioMed
- TMP_DIR                 : Temporary directory
- MQTT_BROKER             : MQTT broker IP address
- MQTT_BROKER_PORT        : MQTT broker port
- UPLOADS_URL             : Upload URL for file repository
'''

import configparser
import os
import uuid

from typing import Any

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.logger import logger
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.constants import ComponentType, HashingAlgorithms


class Environ(metaclass=SingletonMeta):
    """Singleton class contains all variables for researcher or node"""

    def __init__(self, component: ComponentType = None, rootdir: str = None):
        """Class constructor

        Args:
            component: Type of the component either `ComponentType.NODE` or `ComponentType.RESEARCHER`
            rootdir: if not provided the directory is deduced from the package location
                (mainly used by the test files)

        Raises:
            FedbiomedEnvironError: If component type is invalid
        """
        # dict with contains all configuration values
        self._values = {}

        if component == ComponentType.NODE or component == ComponentType.RESEARCHER:
            self._values['COMPONENT_TYPE'] = component
        else:
            _msg = ErrorNumbers.FB600.value + ": parameter should be of ComponentType"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        # common values for all components
        self._init_common(rootdir=rootdir)

        # specific configuration values
        if component == ComponentType.RESEARCHER:
            logger.setLevel("DEBUG")
            self._init_researcher()

        if component == ComponentType.NODE:
            logger.setLevel("INFO")
            self._init_node()

        # display some information on the present environment
        self.info()

    def __getitem__(self, key: str):
        """Override the `[]` get operator to control the Exception type
        Args:
            key: The key of  environ variable

        Raises:
            FedbiomedEnvironError: If the key does not exist
        """
        if key not in self._values:
            _msg = ErrorNumbers.FB600.value + ": config file does not contain the key: " + str(key)
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)
        return self._values[key]

    def __setitem__(self, key: str, value: Any):
        """Override the `[] `set operator to control the Exception type

        Args:
            key: key
            value: value

        Raises:
             FedbiomedEnvironError: If key the does not exist
        """

        if value is None:
            _msg = ErrorNumbers.FB600.value + ": cannot set value to None for key: " + str(key)
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values[key] = value
        return value

    def _init_common(self, rootdir: str):
        """Common configuration values for researcher and node

        Args:
            rootdir: Root directory of Fed-BioMed

        Raises:
             FedbiomedEnvironError: In case of error (OS errors usually)
        """

        # guess the fedbiomed package top dir if no root dir is given
        if rootdir is None:
            # locate the top dir from the file location (got up twice)
            ROOT_DIR = os.path.abspath(
                os.path.join(
                    os.path.dirname(
                        os.path.abspath(__file__)),
                    '..',
                    '..'
                )
            )
        else:
            ROOT_DIR = rootdir

        # Initialize all environment values
        self._values['ROOT_DIR'] = ROOT_DIR

        # main directories
        self._values['CONFIG_DIR'] = os.path.join(ROOT_DIR, 'etc')
        VAR_DIR = os.path.join(ROOT_DIR, 'var')
        self._values['VAR_DIR'] = VAR_DIR
        self._values['CACHE_DIR'] = os.path.join(VAR_DIR, 'cache')
        self._values['TMP_DIR'] = os.path.join(VAR_DIR, 'tmp')

        for _key in 'CONFIG_DIR', 'VAR_DIR', 'CACHE_DIR', 'TMP_DIR':
            dir = self._values[_key]
            if not os.path.isdir(dir):
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    _msg = ErrorNumbers.FB600.value + ": path already exists but is not a directory: " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)
                except OSError:
                    _msg = ErrorNumbers.FB600.value + ": cannot create environment subtree in: " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)

        pass

    def _init_researcher(self):
        """Specific configuration values for researcher

        Raises:
            FedbiomedEnvironError: - if file parser cannot be initialized
                - in case of file system errors (cannot create experiment directories)
        """
        # Parse config file
        cfg = self._parse_config_file()

        # Initialize network configurations for Researcher component
        self._init_network_configurations(cfg)

        # we may remove RESEARCHER_ID in the future (to simplify the code)
        # and use ID instead
        try:
            _cfg_value = cfg.get('default', 'researcher_id')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + \
                   ": no default/researcher_id in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['RESEARCHER_ID'] = os.getenv('RESEARCHER_ID',
                                                  _cfg_value)
        self._values['ID'] = self._values['RESEARCHER_ID']

        ROOT_DIR = self._values['ROOT_DIR']
        VAR_DIR = self._values['VAR_DIR']

        # more directories
        self._values['TENSORBOARD_RESULTS_DIR'] = os.path.join(ROOT_DIR, 'runs')
        self._values['EXPERIMENTS_DIR'] = os.path.join(VAR_DIR, "experiments")

        for _key in 'TENSORBOARD_RESULTS_DIR', 'EXPERIMENTS_DIR':
            dir = self._values[_key]
            if not os.path.isdir(dir):
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    _msg = ErrorNumbers.FB600.value + ": path already exists but is not a directory " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)
                except OSError:
                    _msg = ErrorNumbers.FB600.value + ": cannot create environment subtree in: " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(VAR_DIR, 'queue_messages')

        pass

    def _init_node(self):
        """Specific configuration values for node

        Raises:
            FedbiomedEnvironError: - if file parser cannot be initialized
                - in case of missing keys in the config file
        """

        # Parse config file
        cfg = self._parse_config_file()
        self._init_network_configurations(cfg)

        try:
            _cfg_value = cfg.get('default', 'node_id')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": no default/node_id in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['NODE_ID'] = os.getenv('NODE_ID', _cfg_value)
        self._values['ID'] = self._values['NODE_ID']

        VAR_DIR = self._values['VAR_DIR']
        NODE_ID = self._values['NODE_ID']
        ROOT_DIR = self._values['ROOT_DIR']

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(VAR_DIR,
                                                          f'queue_manager_{NODE_ID}')
        self._values['DB_PATH'] = os.path.join(VAR_DIR,
                                               f'db_{NODE_ID}.json')

        self._values['DEFAULT_TRAINING_PLANS_DIR'] = os.path.join(ROOT_DIR,
                                                          'envs', 'common', 'default_training_plans')

        # default directory for saving training plans that are approved / waiting for approval / rejected
        self._values['TRAINING_PLANS_DIR'] = os.path.join(VAR_DIR, f'training_plans_{NODE_ID}')
        # FIXME: we may want to change that
        # Catch exceptions
        if not os.path.isdir(self._values['TRAINING_PLANS_DIR'] ):
            # create training plan directory
            os.mkdir(self._values['TRAINING_PLANS_DIR'])
        try:
            _cfg_value = cfg.get('security', 'allow_default_training_plans')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + \
                   ": no security/allow_default_training_plans in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['ALLOW_DEFAULT_TRAINING_PLANS'] = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS',
                                                         _cfg_value) \
                                                   .lower() in ('true', '1', 't', True)

        try:
            _cfg_value = cfg.get('security', 'training_plan_approval')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + \
                   ": no security/training_plan_approval in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['TRAINING_PLAN_APPROVAL'] = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL',
                                                   _cfg_value) \
                                             .lower() in ('true', '1', 't', True)

        try:
            _cfg_value = cfg.get('security', 'hashing_algorithm')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + \
                   ": no security/hashing_algorithm in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        hashing_algorithm = _cfg_value

        if hashing_algorithm in HashingAlgorithms.list():
            self._values['HASHING_ALGORITHM'] = hashing_algorithm
        else:
            _msg = ErrorNumbers.FB600.value + ": unknown hashing algorithm: " + str(hashing_algorithm)
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['EDITOR'] = os.getenv('EDITOR')

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

    def _parse_config_file(self):
        """ Read the .ini file corresponding to the component create the file with default values if it does not exist.

        Complete/modify the environment with values coming from the OS environment variables
        """

        # Get config file, it create new config if there is not any
        # get config file location from environment
        # or use a predefined value
        if os.getenv('CONFIG_FILE'):
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

        # Parser for the .ini file
        try:
            cfg = configparser.ConfigParser()
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": cannot parse configuration file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        if os.path.isfile(CONFIG_FILE):
            # get values from .ini file
            try:
                cfg.read(CONFIG_FILE)
            except configparser.Error:
                _msg = ErrorNumbers.FB600.value + ": cannot read config file, check file permissions"
                logger.critical(_msg)
                raise FedbiomedEnvironError(_msg)

        else:
            if self._values['COMPONENT_TYPE'] == ComponentType.RESEARCHER:
                self._create_researcher_config_file(cfg, CONFIG_FILE)
            else:
                self._create_node_config_file(cfg, CONFIG_FILE)

        # store the CONFIG_FILE in environ (may help to debug)
        self._values['CONFIG_FILE'] = CONFIG_FILE

        return cfg

    def _create_node_config_file(self, cfg: dict, config_file: str):
        """Creates new config file for node

        Args:
            cfg: Config object
            config_file: The path indicated where config file
                should be saved.
        """

        # get uploads url
        uploads_url = self._get_uploads_url()

        # TODO: We may remove node_id in the future (to simplify the code)
        node_id = os.getenv('NODE_ID', 'node_' + str(uuid.uuid4()))

        cfg['default'] = {
            'node_id': node_id,
            'uploads_url': uploads_url
        }

        # Message broker
        mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        mqtt_broker_port = int(os.getenv('MQTT_BROKER_PORT', 1883))

        cfg['mqtt'] = {
            'broker_ip': mqtt_broker,
            'port': mqtt_broker_port,
            'keep_alive': 60
        }

        # Security variables
        # Default hashing algorithm is SHA256
        allow_default_training_plans = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', True)
        training_plan_approval = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', False)

        cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': allow_default_training_plans,
            'training_plan_approval': training_plan_approval
        }

        # write the config for future relaunch of the same component
        # (only if the file does not exist)
        try:
            with open(config_file, 'w') as f:
                cfg.write(f)
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": cannot save config file: " + config_file
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        pass

    def _create_researcher_config_file(self, cfg: dict, config_file: str):
        """Create config file for researcher

        Args:
            cfg: Config object
            config_file: The path indicated where config file
                should be saved.
        Raises:
             FedbiomedEnvironError: If config file cannot be created
        """

        # get uploads url
        uploads_url = self._get_uploads_url()

        # Default configuration
        researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))
        cfg['default'] = {
            'researcher_id': researcher_id,
            'uploads_url': uploads_url
        }

        # Message broker
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
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": cannot save config file: " + config_file
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

    def _init_network_configurations(self, cfg: dict):
        """Initialize network configurations
        Args:
            cfg: Config object

        Raises:
            FedbiomedEnvironError: In case of missing keys/values
        """

        # broker location
        try:
            _cfg_value = cfg.get('mqtt', 'broker_ip')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": no mqtt/broker_ip in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['MQTT_BROKER'] = os.getenv('MQTT_BROKER',
                                                _cfg_value)

        try:
            _cfg_value = cfg.get('mqtt', 'port')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": no mqtt/port in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['MQTT_BROKER_PORT'] = int(os.getenv('MQTT_BROKER_PORT',
                                                         _cfg_value))

        # repository location
        try:
            _cfg_value = cfg.get('default', 'uploads_url')
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + \
                   ": no default/uploads_url in config file, please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        UPLOADS_URL = _cfg_value
        uploads_ip = os.getenv('UPLOADS_IP')
        if uploads_ip:
            UPLOADS_URL = "http://" + uploads_ip + ":8844/upload/"

        UPLOADS_URL = os.getenv('UPLOADS_URL', UPLOADS_URL)

        # trailing slash is needed for repo url
        if not UPLOADS_URL.endswith('/'):
            UPLOADS_URL += '/'

        self._values['UPLOADS_URL'] = UPLOADS_URL
        self._values['TIMEOUT'] = 5

    @staticmethod
    def _get_uploads_url() -> str:
        """Gets uploads url from env

        Returns:
            Uploads url
        """

        # use default values from current OS environment variables
        # repository location
        uploads_url = "http://localhost:8844/upload/"
        uploads_ip = os.getenv('UPLOADS_IP')
        if uploads_ip:
            uploads_url = "http://" + uploads_ip + ":8844/upload/"
        uploads_url = os.getenv('UPLOADS_URL', uploads_url)

        return uploads_url

    def info(self):
        """Print useful information at environment creation"""

        logger.info("Component environment:")
        if self._values['COMPONENT_TYPE'] == ComponentType.RESEARCHER:
            logger.info("type = " + str(self._values['COMPONENT_TYPE']))

        if self._values['COMPONENT_TYPE'] == ComponentType.NODE:
            logger.info("type                = " + str(self._values['COMPONENT_TYPE']))
            logger.info("training_plan_approval      = " + str(self._values['TRAINING_PLAN_APPROVAL']))
            logger.info("allow_default_training_plans = " + str(self._values['ALLOW_DEFAULT_TRAINING_PLANS']))
