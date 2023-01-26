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
- MPSPDZ_IP               : MPSPDZ endpoint IP of component
'''

import configparser
import os

from abc import abstractmethod
from typing import Any, Tuple, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedEnvironError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.singleton import SingletonABCMeta
from fedbiomed.common.constants import MPSPDZ_certificate_prefix
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

    def __getitem__(self, key: str) -> Any:
        """Override the `[]` get operator to control the Exception type
        Args:
            key: The key of environ variable

        Returns:
            The value of the environ variable

        Raises:
            FedbiomedEnvironError: If the key does not exist
        """
        if key not in self._values:
            _msg = ErrorNumbers.FB600.value + ": config file does not contain the key: " + str(key)
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> Any:
        """Override the `[] `set operator to control the Exception type

        Args:
            key: key
            value: value

        Returns:
            The value passed as argument

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

    def from_config(self, section, key) -> Any:
        """Gets values from config file
        Args:
            section: the section of the key
            key: the name of the key

        Returns:
            The value of the key

        Raises:
            FedbiomedEnvironError: If the key does not exist in the configuration
        """
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

        Args:
            new: True if configuration file is not expected to exist

        Raise:
            FedbiomedEnvironError: cannot read configuration file
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

        public_key = self.from_config("mpspdz", "public_key")
        private_key = self.from_config("mpspdz", "private_key")
        self._values["MPSPDZ_CERTIFICATE_KEY"] = os.getenv(
            "MPSPDZ_CERTIFICATE_KEY",
            os.path.join(self._values["CONFIG_DIR"], private_key)
        )
        self._values["MPSPDZ_CERTIFICATE_PEM"] = os.getenv(
            "MPSPDZ_CERTIFICATE_PEM",
            os.path.join(self._values["CONFIG_DIR"], public_key)
        )

    def _get_uploads_url(self,
                         from_config: bool = False
                         ) -> str:
        """Gets uploads url from env

        # TODO: Get IP, port and end-point information separately

        Args:
            from_config: if True, use uploads URL value from config as default value, if False use last resort
                default value.

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
            component_id
    ) -> Tuple[str, str]:
        """Generates certificates

        Args:
            component_id: ID of the component for which the certificate will be generated

        Returns:
            key_file: The path where private key file is saved
            pem_file: The path where public key file is saved

        Raises:
            FedbiomedEnvironError: If certificate directory for the component has already `certificate.pem` or
                `certificate.key` files generated.
        """

        certificate_path = os.path.join(self._values["CERT_DIR"], f"cert_{component_id}")

        if os.path.isdir(certificate_path) \
                and (os.path.isfile(os.path.join(certificate_path, "certificate.key")) or
                     os.path.isfile(os.path.join(certificate_path, "certificate.pem"))):

            raise FedbiomedEnvironError(f"Certificate generation is aborted. Directory {certificate_path} already "
                                        f"certificates. Please remove those files to regenerate")
        else:
            os.makedirs(certificate_path, exist_ok=True)

        try:
            key_file, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
                certificate_folder=certificate_path,
                certificate_name=MPSPDZ_certificate_prefix,
                component_id=component_id
            )
        except FedbiomedError as e:
            raise FedbiomedEnvironError(f"Can not generate certificate: {e}")

        return key_file, pem_file

    def _configure_secure_aggregation(self):
        """ Add MPSDPZ section into configuration file."""

        ip, port = self._retrieve_ip_and_port(self._values["PORT_INCREMENT_FILE"])

        component_id = self.from_config("default", "id")

        # Generate self-signed certificates
        key_file, pem_file = self._generate_certificate(
            component_id=component_id
        )

        self._cfg['mpspdz'] = {
            'private_key': os.path.relpath(key_file, self._values["CONFIG_DIR"]),
            'public_key': os.path.relpath(pem_file, self._values["CONFIG_DIR"]),
            'mpspdz_ip': ip,
            'mpspdz_port': port
        }

    @staticmethod
    def _retrieve_ip_and_port(
            increment_file,
            new: bool = False,
            increment: Union[int, None] = None
    ) -> Tuple[str, int]:
        """Creates MPSDPZ IP and PORT based on increment file for ports

        Args:
            increment_file: Path to port increment file
            new: If `True`, ignores increment file and create new one
            increment: if not None, increment value (port number) that will be used if `new = True`.
                If None, then use a default value (environment variable or last resort value)

        Returns:
            ip: The IP for the MPSDPZ
            port: The port number for MPSPDZ
        """

        ip = os.getenv('MPSPDZ_IP', "localhost")

        if os.path.isfile(increment_file) and new is False:
            with open(increment_file, "r+") as file:
                port_increment = file.read()
                if port_increment != "":
                    port = int(port_increment) + 1
                    file.truncate(0)
                    file.close()

                    # Renew port in the  file
                    _ = Environ._retrieve_ip_and_port(
                        increment_file,
                        new=True,
                        increment=port)
                else:
                    _, port = Environ._retrieve_ip_and_port(
                        increment_file,
                        new=True)
        else:
            with open(increment_file, "w") as file:
                port = os.getenv('MPSPDZ_PORT', 14000) if increment is None else increment
                file.write(f"{port}")
                file.close()

        return ip, port

    def _write_config_file(self):

        # write the config for future relaunch of the same component
        # (only if the file does not exist)

        if "CONFIG_FILE" not in self._values:
            raise FedbiomedEnvironError("Please set config file first!")

        try:
            with open(self._values["CONFIG_FILE"], 'w') as f:
                self._cfg.write(f)
        except configparser.Error:
            _msg = ErrorNumbers.FB600.value + ": cannot save config file: " + self._values["CONFIG_FILE"]
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)
