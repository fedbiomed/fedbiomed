# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import configparser
import os 
import uuid 

from abc import ABCMeta, abstractmethod

from typing import Optional

from fedbiomed.common.constants import ErrorNumbers, MPSPDZ_certificate_prefix, \
    SERVER_certificate_prefix, \
    __researcher_config_version__, __node_config_version__, \
    HashingAlgorithms, \
    CONFIG_FOLDER_NAME, \
    VAR_FOLDER_NAME, \
    DB_PREFIX
from fedbiomed.common.utils import (
    create_fedbiomed_setup_folders,
    raise_for_version_compatibility,
    CONFIG_DIR,
    ROOT_DIR)
from fedbiomed.common.certificate_manager import retrieve_ip_and_port, generate_certificate


class Config(metaclass=ABCMeta):
    """Base Config class"""

    DEFAULT_CONFIG_FILE_NAME: str = 'config'
    COMPONENT_TYPE: str

    def __init__(
        self,
        root = None,
        name: Optional[str] = None,
        auto_generate: bool = True
    ) -> None:
        """Initializes config"""
        self.root = root
        self._cfg = configparser.ConfigParser()
        self.name = name if name \
            else os.getenv("CONFIG_FILE", self.DEFAULT_CONFIG_FILE_NAME)

        if self.root:
            self.path = os.path.join(self.root, CONFIG_FOLDER_NAME, self.name)
            self.root = self.root
        else:
            self.path = os.path.join(CONFIG_DIR, self.name)
            self.root = ROOT_DIR

        # Creates setup folders if not existing
        create_fedbiomed_setup_folders(self.root)

        if auto_generate:
            self.generate()


    def is_config_existing(self) -> bool:
        """Checks if config file is exsiting

        Returns:
            True if config file is already existing
        """

        return os.path.isfile(self.path)

    def read(self) -> bool:
        """Reads configuration file that is already existing in given path

        Raises verision compatibility error
        """

        self._cfg.read(self.path)

        # Validate config version
        raise_for_version_compatibility(
            self._cfg["default"]["version"],
            self.CONFIG_VERSION,
            f"Configuration file {self.path}: found version %s expected version %s")

        return True

    def get(self, section, key) -> str:
        """Returns value for given ket and section"""

        return self._cfg.get(section, key)

    def sections(self) -> list:
        """Returns sections of the config"""

        return self._cfg.sections()

    def write(self):
        """Writes config file"""

        try:
            with open(self.path, 'w') as f:
                self._cfg.write(f)
        except configparser.Error:
            raise IOError(ErrorNumbers.FB600.value + ": cannot save config file: " + self.path)


    def generate(self, force: bool = False) -> bool:
        """"Generate configuration file

        Args:
        force: Overwrites existing configration file
        """

        # Check if configuration is already existing
        if self.is_config_existing() and not force:
            return self.read()

        component_id = f"{self.COMPONENT_TYPE}_{uuid.uuid4()}"

        self._cfg['default'] = {
            'id': component_id,
            'component': self.COMPONENT_TYPE,
            'version': str(self.CONFIG_VERSION)
        }

        # DB PATH RELATIVE
        db_path  = os.path.join(self.root, VAR_FOLDER_NAME, f"{DB_PREFIX}{component_id}.json")
        self._cfg['default']['db'] = os.path.relpath(db_path, os.path.join(self.root, CONFIG_FOLDER_NAME))

        ip, port = retrieve_ip_and_port(self.root)
        allow_default_biprimes = os.getenv('ALLOW_DEFAULT_BIPRIMES', True)

        # Generate self-signed certificates
        key_file, pem_file = generate_certificate(
            root=self.root, 
            component_id=component_id, 
            prefix=MPSPDZ_certificate_prefix)

        self._cfg['mpspdz'] = {
            'private_key': os.path.relpath(key_file, os.path.join(self.root, 'etc')),
            'public_key': os.path.relpath(pem_file, os.path.join(self.root, 'etc')),
            'mpspdz_ip': ip,
            'mpspdz_port': port,
            'allow_default_biprimes': allow_default_biprimes,
            'default_biprimes_dir': os.path.relpath(
                os.path.join(self.root, 'envs', 'common', 'default_biprimes'),
                os.path.join(self.root, 'etc')
            )
        }

        # Calls child class add_parameterss
        self.add_parameters()

        # Write configuration file
        return self.write()

    @abstractmethod
    def add_parameters(self):
        """"Component specific argument creation"""


class NodeConfig(Config):

    DEFAULT_CONFIG_FILE_NAME: str = 'config_node.ini'
    COMPONENT_TYPE: str = 'NODE'
    CONFIG_VERSION: str = __node_config_version__

    def add_parameters(self):
        """Generate researcher config"""

        # Security variables
        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', True),
            'training_plan_approval': os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', False),
            'secure_aggregation': os.getenv('SECURE_AGGREGATION', True),
            'force_secure_aggregation': os.getenv('FORCE_SECURE_AGGREGATION', False)
        }

        # gRPC server host and port
        self._cfg["researcher"] = {
            'ip': os.getenv('RESEARCHER_SERVER_HOST', 'localhost'),
            'port': os.getenv('RESEARCHER_SERVER_PORT', '50051')
        }


class ResearcherConfig(Config):

    DEFAULT_CONFIG_FILE_NAME: str = 'config_researcher.ini'
    COMPONENT_TYPE: str = 'RESEARCHER'
    CONFIG_VERSION: str = __researcher_config_version__

    def add_parameters(self):
        """Generate researcher config"""

        grpc_host = os.getenv('RESEARCHER_SERVER_HOST', 'localhost')
        grpc_port = os.getenv('RESEARCHER_SERVER_PORT', '50051')

        # Generate certificate for gRPC server
        key_file, pem_file = generate_certificate(
            root=self.root, 
            component_id=self._cfg['default']['id'],
            prefix=SERVER_certificate_prefix,
            subject={'CommonName': grpc_host}
        )

        self._cfg['server'] = {
            'host': grpc_host,
            'port': grpc_port,
            'pem' : os.path.relpath(pem_file, os.path.join(self.root, CONFIG_FOLDER_NAME)),
            'key' : os.path.relpath(key_file, os.path.join(self.root, CONFIG_FOLDER_NAME))
        }
