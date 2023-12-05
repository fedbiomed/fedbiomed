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
- MPSPDZ_IP               : MP-SPDZ endpoint IP of component
- DEFAULT_BIPRIMES_DIR    : Path of directory for storing default secure aggregation biprimes
- ALLOW_DEFAULT_BIPRIMES  : True if the component enables the default secure aggregation biprimes
- PORT_INCREMENT_FILE     : File for storing next port to be allocated for MP-SPDZ
- CERT_DIR                : Directory for storing certificates for MP-SPDZ
- DEFAULT_BIPRIMES_DIR    : Directory for storing default biprimes files
'''

import os

from typing import Any

from fedbiomed.common.constants import ErrorNumbers, VAR_FOLDER_NAME, \
    CACHE_FOLDER_NAME, CONFIG_FOLDER_NAME, TMP_FOLDER_NAME, \
    CERTS_FOLDER_NAME
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.utils import (
    ROOT_DIR, 
    CONFIG_DIR, 
    VAR_DIR, 
    CACHE_DIR, 
    TMP_DIR,
)
from fedbiomed.common.logger import logger
from fedbiomed.common.singleton import SingletonABCMeta


class Environ(metaclass=SingletonABCMeta):
    """Singleton class contains all variables for researcher or node"""

    def __init__(self, root_dir: str = None):
        """Class constructor

        Args:
            root_dir: if not provided the directory is deduced from the package location
                (specifying root_dir is mainly used by the test files)

        Raises:
            FedbiomedEnvironError: If component type is invalid
        """
        # dict with contains all configuration values
        self._values = {}
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

    def set_environment(self):
        """Common configuration values for researcher and node

        Raises:
             FedbiomedEnvironError: In case of error (OS errors usually)
        """

        # guess the fedbiomed package top dir if no root dir is given
        if self._root_dir is None:
            root_dir = ROOT_DIR

            # initialize main directories
            self._values['ROOT_DIR'] = root_dir
            self._values['CONFIG_DIR'] = CONFIG_DIR
            self._values['VAR_DIR'] = VAR_DIR
            self._values['CACHE_DIR'] = CACHE_DIR
            self._values['TMP_DIR'] = TMP_DIR
        else:
            root_dir = self._root_dir
            # initialize main directories

            self._values['ROOT_DIR'] = root_dir
            self._values['CONFIG_DIR'] = os.path.join(root_dir, CONFIG_FOLDER_NAME)
            self._values['VAR_DIR'] = os.path.join(root_dir, VAR_FOLDER_NAME)
            self._values['CACHE_DIR'] = os.path.join(self._values['VAR_DIR'], CACHE_FOLDER_NAME)
            self._values['TMP_DIR'] = os.path.join(self._values['VAR_DIR'], TMP_FOLDER_NAME)


        self._values['DB_PATH'] = os.path.normpath(
            os.path.join(self._values["ROOT_DIR"], CONFIG_FOLDER_NAME, self.config.get('default', 'db'))
        )

        # initialize other directories
        self._values['PORT_INCREMENT_FILE'] = os.path.join(root_dir, CONFIG_FOLDER_NAME, "port_increment")
        self._values['CERT_DIR'] = os.path.join(root_dir, CERTS_FOLDER_NAME)
        self._values['DEFAULT_BIPRIMES_DIR'] = os.path.join(root_dir, 'envs', 'common', 'default_biprimes')


        self._values['TIMEOUT'] = 5
        self._values["MPSPDZ_IP"] = os.getenv("MPSPDZ_IP", 
                                              self.config.get("mpspdz", "mpspdz_ip"))
        self._values["MPSPDZ_PORT"] = os.getenv("MPSPDZ_PORT", 
                                                self.config.get("mpspdz", "mpspdz_port"))

        self._values['ALLOW_DEFAULT_BIPRIMES'] = \
            os.getenv('ALLOW_DEFAULT_BIPRIMES', 
                      self.config.get('mpspdz', 'allow_default_biprimes') ) \
            .lower() in ('true', '1', 't', True)

        public_key = self.config.get("mpspdz", "public_key")
        private_key = self.config.get("mpspdz", "private_key")
        self._values["MPSPDZ_CERTIFICATE_KEY"] = os.getenv(
            "MPSPDZ_CERTIFICATE_KEY",
            os.path.join(self._values["CONFIG_DIR"], private_key)
        )
        self._values["MPSPDZ_CERTIFICATE_PEM"] = os.getenv(
            "MPSPDZ_CERTIFICATE_PEM",
            os.path.join(self._values["CONFIG_DIR"], public_key)
        )
