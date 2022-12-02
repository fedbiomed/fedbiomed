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

from abc import ABC, abstractmethod
from typing import Any, Union, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedEnvironError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.singleton import SingletonABCMeta
from fedbiomed.common.constants import ComponentType, HashingAlgorithms
from fedbiomed.common.certificate_manager import CertificateManager


class Environ(metaclass=SingletonABCMeta):
    """Singleton class contains all variables for researcher or node"""

    def __init__(self, root_dir: str = None):
        """Class constructor

        Args:
            root_dir: if not provided the directory is deduced from the package location
                (mainly used by the test files)

        Raises:
            FedbiomedEnvironError: If component type is invalid
        """
        # dict with contains all configuration values
        self._values = {}
        self._cfg = configparser.ConfigParser()
        self._root_dir = root_dir

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

    @abstractmethod
    def _set_component_specific_variables(self):
        """Abstract method for setting component specific values to `self._values` """

    @abstractmethod
    def _set_component_specific_config_parameters(self):
        """Abstract method for setting component specific parameters"""

    @abstractmethod
    def default_config_file(self) -> str:
        """Abstract method for retrieving default configuration file path"""

    @abstractmethod
    def info(self):
        """Abstract method to return component information"""

    def from_config(self, section, key):
        """Gets values from config file"""
        try:
            _cfg_value = self._cfg.get(section, key)
        except configparser.Error:
            _msg = f"{ErrorNumbers.FB600.value}: no {section}/{key} in config file. Please recreate a new config file"
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        return _cfg_value

    def setup_environment(self):
        """Final environment setup function """
        # Initialize common environment variables
        self._initialize_common_variables()

        # Parse config file or create if not existing
        self.parse_write_config_file()

        # Configuring network variables
        self._set_network_variables()

        # Initialize environment variables
        self._set_component_specific_variables()

    def _initialize_common_variables(self):
        """Common configuration values for researcher and node

        Raises:
             FedbiomedEnvironError: In case of error (OS errors usually)
        """

        # guess the fedbiomed package top dir if no root dir is given
        if self._root_dir is None:
            # locate the top dir from the file location (got up twice)
            root_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(
                        os.path.abspath(__file__)),
                    '..',
                    '..'
                )
            )
        else:
            root_dir = self._root_dir

        # Initialize all environment values
        self._values['ROOT_DIR'] = root_dir

        # main directories
        self._values['CONFIG_DIR'] = os.path.join(root_dir, 'etc')
        self._values['VAR_DIR'] = os.path.join(root_dir, 'var')
        self._values['CACHE_DIR'] = os.path.join(self._values['VAR_DIR'], 'cache')
        self._values['TMP_DIR'] = os.path.join(self._values['VAR_DIR'], 'tmp')
        self._values['PORT_INCREMENT_FILE'] = os.path.join(root_dir, "etc", "port_increment")
        self._values['CERT_DIR'] = os.path.join(root_dir, "etc", "certs")

        for _key in 'CONFIG_DIR', 'VAR_DIR', 'CACHE_DIR', 'TMP_DIR', 'CERT_DIR':
            dir_ = self._values[_key]
            if not os.path.isdir(dir_):
                try:
                    os.makedirs(dir_)
                except FileExistsError:
                    _msg = ErrorNumbers.FB600.value + ": path already exists but is not a directory: " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)
                except OSError:
                    _msg = ErrorNumbers.FB600.value + ": cannot create environment subtree in: " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)

        pass

    def set_config_file(self):
        """Sets configuration file """
        config_file = os.getenv('CONFIG_FILE')
        if config_file:
            if not os.path.isabs(config_file):
                config_file = os.path.join(self._values['CONFIG_DIR'],
                                           os.getenv('CONFIG_FILE'))
        else:
            config_file = self.default_config_file()

        self._values["CONFIG_FILE"] = config_file

    def parse_write_config_file(self, new: bool = False):
        """Parses configuration file.

        Create new config file if it is not existing.
        """
        # Sets configuration file path
        self.set_config_file()

        # Parse configuration if it is existing
        if os.path.isfile(self._values["CONFIG_FILE"]) and not new:
            # get values from .ini file
            try:
                self._cfg.read(self._values["CONFIG_FILE"])
            except configparser.Error:
                _msg = ErrorNumbers.FB600.value + ": cannot read config file, check file permissions"
                logger.critical(_msg)
                raise FedbiomedEnvironError(_msg)

        # Create new configuration file
        else:
            # Create new config file
            self._set_component_specific_config_parameters()

            # Updates config file with MQTT configuration
            self._configure_mqtt()

            # Update config with secure aggregation parameters
            self._configure_secure_aggregation()

            # Writes config file to a file
            self._write_config_file()

    def _set_network_variables(self):
        """Initialize network configurations

        Raises:
            FedbiomedEnvironError: In case of missing keys/values
        """

        # broker location
        broker_ip = self.from_config('mqtt', 'broker_ip')
        broker_port = self.from_config('mqtt', 'port')
        self._values['MQTT_BROKER'] = os.getenv('MQTT_BROKER', broker_ip)
        self._values['MQTT_BROKER_PORT'] = int(os.getenv('MQTT_BROKER_PORT', broker_port))

        # Uploads URL
        uploads_url = self._get_uploads_url(from_config=True)

        self._values['UPLOADS_URL'] = uploads_url
        self._values['TIMEOUT'] = 5

        # MPSPDZ variables
        mpspdz_ip = self.from_config("mpspdz", "mpspdz_ip")
        mpspdz_port = self.from_config("mpspdz", "mpspdz_port")
        self._values["MPSPDZ_IP"] = os.getenv("MPSPDZ_IP", mpspdz_ip)
        self._values["MPSPDZ_PORT"] = os.getenv("MPSPDZ_PORT", mpspdz_port)

        public_key = self.from_config("ssl", "public_key")
        private_key = self.from_config("ssl", "private_key")
        self._values["CERTIFICATE_KEY"] = os.getenv("SLL_PRIVATE_KEY", private_key)
        self._values["CERTIFICATE_PEM"] = os.getenv("SLL_PUBLIC_KEY", public_key)

    def _get_uploads_url(self,
                         from_config: Union[None, str] = False
                         ) -> str:
        """Gets uploads url from env

        # TODO: Get IP, port and end-point information separately

        Returns:
            Uploads url
        """

        uploads_url = self.from_config("default", "uploads_url") if \
            from_config is True else \
            "http://localhost:8844/upload/"

        # Modify URL with custom IP
        uploads_ip = os.getenv('UPLOADS_IP')
        if uploads_ip:
            uploads_url = f"http://{uploads_ip}:8844/upload/"

        # Environment variable always overwrites config value
        url = os.getenv('UPLOADS_URL', uploads_url)

        return url

    def _configure_mqtt(self):
        """Configures MQTT  credentials."""

        # Message broker
        mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        mqtt_broker_port = int(os.getenv('MQTT_BROKER_PORT', 1883))

        self._cfg['mqtt'] = {
            'broker_ip': mqtt_broker,
            'port': mqtt_broker_port,
            'keep_alive': 60
        }

    def _generate_certificate(
            self,
            component_id,
            certificate_data: dict = {}
    ) -> Tuple[str, str]:
        """Generates certificates

        Args:
            component_id: ID of the component for which the certificate will be generated
            certificate_data: Data for certificates to declare, `email`, `country`, `organization`, `validity`.
                Certificate data should be dict where `email`, `country`, `organization` is string type and `validity`
                boolean

        Raises:
            FedbiomedEnvironError: If certificate directory for the component has already `certificate.pem` or
                `certificate.key` files generated.

        Returns:
            key_file: The path where private key file is saved
            pem_file: The path where public key file is saved
        """

        certificate_path = os.path.join(self._values["CERT_DIR"], f"cert_{component_id}")

        if os.path.isdir(certificate_path) \
                and (os.path.isfile(os.path.join(certificate_path, "certificate.key"))
                     or os.path.isfile(os.path.join(certificate_path, "certificate.pem"))):

            raise FedbiomedEnvironError(f"Certificate generation is aborted. Directory {certificate_path} already "
                                        f"certificates. Please remove those files to regenerate")
        else:
            os.makedirs(certificate_path)

        try:
            key_file, pem_file = CertificateManager.generate_certificate(
                certificate_path,
                certificate_data
            )
        except FedbiomedError as e:
            raise FedbiomedEnvironError(f"Can not generate certificate: {e}")

        return key_file, pem_file

    def _configure_secure_aggregation(self):
        """ Add MPSDPZ section into configuration file."""

        ip, port = self._retrieve_ip_and_port(self._values["PORT_INCREMENT_FILE"])

        self._cfg['mpspdz'] = {
            'mpspdz_ip': ip,
            'mpspdz_port': port
        }

    @staticmethod
    def _retrieve_ip_and_port(
            increment_file,
            new: bool = False,
            increment: int = None
    ) -> Tuple[str, int]:
        """Creates MPSDPZ IP and PORT based on increment file for ports

        Args:
            increment_file: Path to port increment file
            new: If `True`, ignores increment file and create new one
            increment: Increment value (port number) that will be used if `new = True`.

        Returns:
            ip: The IP for the MPSDPZ
            port: The port number for MPSPDZ
        """

        ip = os.getenv('MPSPDZ_IP', "localhost")

        if os.path.isfile(increment_file) and new is False:
            with open(increment_file, "r+") as file:
                port_increment = file.read()
                if port_increment != "":
                    port = int(port_increment.split(":")[1])
                    file.truncate(0)
                    file.close()

                    # Renew port in the  file
                    _ = Environ._retrieve_ip_and_port(
                        increment_file,
                        new=True,
                        increment=port + 1)
                else:
                    ip, port = Environ._retrieve_ip_and_port(
                        increment_file,
                        new=True)
        else:
            with open(increment_file, "w") as file:
                port = os.getenv('MPSPDZ_PORT', 14000) if increment is None else increment
                file.write(f"{ip}:{port}")
                file.close()

        return ip, port

    def _write_config_file(self):

        # write the config for future relaunch of the same component
        # (only if the file does not exist)

        if "CONFIG_FILE" not in self._values:
            raise FedbiomedEnvironError(f"Please set config file first!")

        try:
            with open(self._values["CONFIG_FILE"], 'w') as f:
                self._cfg.write(f)
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": cannot save config file: " + self._values["CONFIG_FILE"]
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)


class ResearcherEnviron(Environ):

    def __init__(self, root_dir: str = None):
        """Constructs ResearcherEnviron object """
        super().__init__(root_dir=root_dir)
        logger.setLevel("DEBUG")
        self._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER

        # Setup environment variables
        self.setup_environment()

    def default_config_file(self) -> str:
        """Sets config file path """

        return os.path.join(self._values['CONFIG_DIR'], 'config_researcher.ini')

    def _set_component_specific_variables(self):

        # we may remove RESEARCHER_ID in the future (to simplify the code)
        # and use ID instead

        researcher_id = self.from_config('default', 'researcher_id')

        self._values['RESEARCHER_ID'] = os.getenv('RESEARCHER_ID', researcher_id)
        self._values['ID'] = self._values['RESEARCHER_ID']

        # more directories
        self._values['TENSORBOARD_RESULTS_DIR'] = os.path.join(self._values['ROOT_DIR'], 'runs')
        self._values['EXPERIMENTS_DIR'] = os.path.join(self._values['VAR_DIR'], "experiments")
        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(self._values['VAR_DIR'], 'queue_messages')
        self._values['DB_PATH'] = os.path.join(self._values['VAR_DIR'],
                                               f'db_{self._values["RESEARCHER_ID"]}.json')
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

    def _set_component_specific_config_parameters(self):
        # get uploads url
        uploads_url = self._get_uploads_url()

        # Default configuration
        researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))
        self._cfg['default'] = {
            'researcher_id': researcher_id,
            'uploads_url': uploads_url
        }

        # Generate self-signed certificates
        key_file, pem_file = self._generate_certificate(researcher_id)

        # Set public and private keys
        self._cfg['ssl'] = {
            'private_key': key_file,
            'public_key': pem_file
        }

    def info(self):
        """Print useful information at environment creation"""

        logger.info("Component environment:")
        logger.info("type = " + str(self._values['COMPONENT_TYPE']))


class NodeEnviron(Environ):

    def __init__(self, root_dir: str = None):
        """Constructs NodeEnviron object """
        super().__init__(root_dir=root_dir)
        logger.setLevel("INFO")
        self._values["COMPONENT_TYPE"] = ComponentType.NODE
        # Setup environment variables
        self.setup_environment()

    def default_config_file(self) -> str:
        """Sets config file path """

        return os.path.join(self._values['CONFIG_DIR'], 'config_node.ini')

    def _set_component_specific_variables(self):
        """Initializes environment variables """

        node_id = self.from_config('default', 'node_id')
        self._values['NODE_ID'] = os.getenv('NODE_ID', node_id)
        self._values['ID'] = self._values['NODE_ID']

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(self._values['VAR_DIR'],
                                                          f'queue_manager_{self._values["NODE_ID"]}')
        self._values['DB_PATH'] = os.path.join(self._values['VAR_DIR'],
                                               f'db_{self._values["NODE_ID"]}.json')

        self._values['DEFAULT_TRAINING_PLANS_DIR'] = os.path.join(self._values['ROOT_DIR'],
                                                                  'envs', 'common', 'default_training_plans')

        # default directory for saving training plans that are approved / waiting for approval / rejected
        self._values['TRAINING_PLANS_DIR'] = os.path.join(self._values['VAR_DIR'],
                                                          f'training_plans_{self._values["NODE_ID"]}')
        # FIXME: we may want to change that
        # Catch exceptions
        if not os.path.isdir(self._values['TRAINING_PLANS_DIR']):
            # create training plan directory
            os.mkdir(self._values['TRAINING_PLANS_DIR'])

        allow_dtp = self.from_config('security', 'allow_default_training_plans')

        self._values['ALLOW_DEFAULT_TRAINING_PLANS'] = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS',
                                                                 allow_dtp) \
                                                           .lower() in ('true', '1', 't', True)

        tp_approval = self.from_config('security', 'training_plan_approval')

        self._values['TRAINING_PLAN_APPROVAL'] = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL',
                                                           tp_approval) \
                                                     .lower() in ('true', '1', 't', True)

        hashing_algorithm = self.from_config('security', 'hashing_algorithm')
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

    def _set_component_specific_config_parameters(self):
        """Updates config file with Node specific parameters"""

        # TODO: We may remove node_id in the future (to simplify the code)
        node_id = os.getenv('NODE_ID', 'node_' + str(uuid.uuid4()))
        uploads_url = self._get_uploads_url()

        self._cfg['default'] = {
            'node_id': node_id,
            'uploads_url': uploads_url
        }

        # Generate self-signed certificates
        key_file, pem_file = self._generate_certificate(node_id)

        # Security variables
        # Default hashing algorithm is SHA256
        allow_default_training_plans = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', True)
        training_plan_approval = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', False)

        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': allow_default_training_plans,
            'training_plan_approval': training_plan_approval
        }

        # Set public and private keys
        self._cfg['ssl'] = {
            'private_key': key_file,
            'public_key': pem_file
        }

    def info(self):
        """"""
        logger.info("type                           = " + str(self._values['COMPONENT_TYPE']))
        logger.info("training_plan_approval         = " + str(self._values['TRAINING_PLAN_APPROVAL']))
        logger.info("allow_default_training_plans   = " + str(self._values['ALLOW_DEFAULT_TRAINING_PLANS']))
