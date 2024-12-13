# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
import uuid

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Dict

from fedbiomed.common.constants import (
    ErrorNumbers,
    CONFIG_FOLDER_NAME,
    VAR_FOLDER_NAME,
    CERTS_FOLDER_NAME,
    DB_PREFIX,
)
from fedbiomed.common.utils import (
    create_fedbiomed_setup_folders,
    raise_for_version_compatibility,
    CONFIG_DIR,
    ROOT_DIR,
)
from fedbiomed.common.exceptions import FedbiomedConfigurationError

# from fedbiomed.common.secagg_manager import SecaggBiprimeManager


class Config(metaclass=ABCMeta):
    """Base Config class

    Attributes:
        root: Root directory of the component.
        name: Config name (e.g config.ini or config-n1.ini).
        path: Absolute path to configuration.
        vars: A dictionary that contains configuration related variables. Such
            as dynamic paths that relies of component root etc.
    """

    _DEFAULT_CONFIG_FILE_NAME: str = "config"
    _COMPONENT_TYPE: str
    _CONFIG_VERSION: str

    _cfg: configparser.ConfigParser
    root: str
    path: str
    name: str

    vars: Dict[str, Any] = {}

    def __init__(
        self,
        root: str | None = None,
        name: Optional[str] = None,
        auto_generate: bool = True
    ) -> None:
        """Initializes configuration

        Args:
            root: Root directory for the component
            name: Component configuration file name (e.g `config-n1.ini`
                corresponds to `<root>/constants.CONFIG_FOLDER_NAME/config-n1.ini`).
            auto_generate: Generated all component files, folder, including
                configuration file.
        """
        # First try to get component specific config file name, then CONFIG_FILE
        default_config = os.getenv(
            f"{self._COMPONENT_TYPE}_CONFIG_FILE",
            os.getenv("CONFIG_FILE", self._DEFAULT_CONFIG_FILE_NAME),
        )

        self._cfg = configparser.ConfigParser()
        self.name = name if name else default_config
        self.load(self.name, root, auto_generate)

    @classmethod
    @abstractmethod
    def _COMPONENT_TYPE(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining component type"""

    @classmethod
    @abstractmethod
    def _CONFIG_VERSION(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining configuration version"""

    def load(
        self,
        name: str,
        root: str | None = None,
        auto_generate: bool = True
    ) -> None:
        """Load configuration from given name and root

        This implementation allows to load configuration after Config class
        is instantiated.

        Args:
            name: Name of the config file
            root: Root directory where component files will be saved
            auto_generate: Generated all component files, folder, including
                configuration file.
        """
        self.name = name

        if root:
            self.path = os.path.join(root, CONFIG_FOLDER_NAME, self.name)
            self.root = root
        else:
            self.path = os.path.join(CONFIG_DIR, self.name)
            self.root = ROOT_DIR

        if auto_generate or self.is_config_existing():
            self.generate()

        # Creates setup folders if not existing
        create_fedbiomed_setup_folders(self.root)

    def is_config_existing(self) -> bool:
        """Checks if config file exists

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
            self._CONFIG_VERSION,
            f"Configuration file {self.path}: found version %s expected version %s",
        )

        return True

    def get(self, section, key, **kwargs) -> str:
        """Returns value for given key and section"""

        return self._get(section, key, **kwargs)

    def getbool(self, section, key, **kwargs) -> bool:
        """Gets boolean value from config"""

        return self._get(section, key, **kwargs).lower() in ('true', '1')


    def _get(self, section, key, **kwargs) -> str:
        """ """
        environ_key = f"FBM_{section.upper()}_{key.upper()}"
        return os.environ.get(environ_key, self._cfg.get(section, key, **kwargs))

    def set(self, section, key, value) -> None:
        """Sets config section values

        Args:
            section: the name of the config file section as defined by the `ini` standard
            key: the name of the attribute to be set
            value: the value of the attribute to be set

        Returns:
            value: the value of the attribute that was just set
        """
        self._cfg.set(section, key, value)

    def sections(self) -> list:
        """Returns sections of the config"""

        return self._cfg.sections()

    def write(self):
        """Writes config file"""

        try:
            with open(self.path, "w", encoding="UTF-8") as f:
                self._cfg.write(f)
        except configparser.Error as exp:
            raise FedbiomedConfigurationError(
                f"{ErrorNumbers.FB600.value}: cannot save config file: " + self.path
            ) from exp

    def generate(
        self,
        force: bool = False,
        id: Optional[str] = None
    ) -> None:
        """ "Generate configuration file

        Args:
            force: Overwrites existing configration file
            id: Component ID
        """

        # Check if configuration is already existing
        if not self.is_config_existing() or force:
            # Create default section
            component_id = id if id else f"{self._COMPONENT_TYPE}_{uuid.uuid4()}"

            self._cfg["default"] = {
                "id": component_id,
                "component": self._COMPONENT_TYPE,
                "version": str(self._CONFIG_VERSION),
            }

            db_path = os.path.join(
                self.root, VAR_FOLDER_NAME, f"{DB_PREFIX}{component_id}.json"
            )
            self._cfg["default"]["db"] = os.path.relpath(
                db_path, os.path.join(self.root, CONFIG_FOLDER_NAME)
            )


            # Calls child class add_parameterss
            self.add_parameters()

            # Write configuration file
            self.write()
        else:
            self.read()

        self._update_vars()

    def _update_vars(self):
        """Updates dynamic variables"""
        # Updates dynamic variables

        self.vars.update({
            'MESSAGES_QUEUE_DIR': os.path.join(self.root, 'queue_messages'),
            'TMP_DIR': os.path.join(self.root, VAR_FOLDER_NAME, 'tmp'),
            'CERT_DIR': os.path.join(self.root, CERTS_FOLDER_NAME)
        })

        os.makedirs(self.vars['TMP_DIR'], exist_ok=True)

    @abstractmethod
    def add_parameters(self):
        """ "Component specific argument creation"""

    def refresh(self):
        """Refreshes config file by recreating all the fields without
        chaning component ID.
        """

        if not self.is_config_existing():
            raise FedbiomedConfigurationError(
                f"{ErrorNumbers.FB600.value}: Can not refresh config file that is not existing"
            )

        # Read the config
        self._cfg.read(self.path)
        id = self._cfg["default"]["id"]

        # Generate by keeping the component ID
        self.generate(force=True, id=id)
