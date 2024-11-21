# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
All environment/configuration variables are provided by the
**Environ** dictionary.

**Environ** is a singleton class, meaning that only an instance of Environ is available.


Description of the common Global Variables:

- COMPONENT_TYPE              : Node or Researcher
- ROOT_DIR                    : Base directory
- CONFIG_DIR                  : Configuration file path
- VAR_DIR                     : Var directory of Fed-BioMed
- CACHE_DIR                   : Cache directory of Fed-BioMed
- TMP_DIR                     : Temporary directory
- DB_PATH                     : TinyDB database path where datasets/training_plans/loading plans are saved
- CERT_DIR                    : Directory for storing certificates for the component.
- SECAGG_INSECURE_VALIDATION  : True if the use of secagg consistency validation is allowed,
                                though it introduces room for honest but curious attack on secagg crypto.
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
from fedbiomed.common.config import Config

class Environ(metaclass=SingletonABCMeta):
    """Singleton class contains all variables for researcher or node"""

    _config: Config

    def __init__(self):
        """Class constructor

        Args:
            root_dir: if not provided the directory is deduced from the package location
                (specifying root_dir is mainly used by the test files)

        Raises:
            FedbiomedEnvironError: If component type is invalid
        """
        # dict with contains all configuration values

        self._values = {}

    @property
    def config(self) -> Config:
        """Returns config object"""
        return self._config

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

        self._values['ROOT_DIR'] =  self._config.root
        self._values['CONFIG_DIR'] = os.path.join(self._config.root, CONFIG_FOLDER_NAME)
        self._values['VAR_DIR'] = os.path.join(self._config.root, VAR_FOLDER_NAME)
        self._values['CACHE_DIR'] = os.path.join(self._values['VAR_DIR'], CACHE_FOLDER_NAME)
        self._values['TMP_DIR'] = os.path.join(self._values['VAR_DIR'], TMP_FOLDER_NAME)


        self._values['DB_PATH'] = os.path.normpath(
            os.path.join(
                self._values["ROOT_DIR"], CONFIG_FOLDER_NAME, self._config.get('default', 'db'))
        )

        self._values['CERT_DIR'] = os.path.join(self._config.root, CERTS_FOLDER_NAME)

        # Optional secagg_insecure_validation optional in config file
        secagg_insecure_validation = self._config.get(
            'security', 'secagg_insecure_validation', fallback='true')
        self._values["SECAGG_INSECURE_VALIDATION"] = os.getenv(
            'SECAGG_INSECURE_VALIDATION',
            secagg_insecure_validation).lower() in ('true', '1', 't', True)
