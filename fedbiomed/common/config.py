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
    DB_PREFIX,
)
from fedbiomed.common.utils import (
    create_fedbiomed_setup_folders,
    raise_for_version_compatibility,
    CONFIG_DIR,
    ROOT_DIR,
)
from fedbiomed.common.certificate_manager import (
    generate_certificate,
)
from fedbiomed.common.exceptions import FedbiomedError

# from fedbiomed.common.secagg_manager import SecaggBiprimeManager


class Config(metaclass=ABCMeta):
    """Base Config class"""

    _DEFAULT_CONFIG_FILE_NAME: str = "config"
    _COMPONENT_TYPE: str
    _CONFIG_VERSION: str

    def __init__(
        self, root=None, name: Optional[str] = None, auto_generate: bool = True
    ) -> None:
        """Initializes config"""

        # First try to get component specific config file name, then CONFIG_FILE
        default_config = os.getenv(
            f"{self._COMPONENT_TYPE}_CONFIG_FILE",
            os.getenv("CONFIG_FILE", self._DEFAULT_CONFIG_FILE_NAME),
        )

        self.root = root
        self._cfg = configparser.ConfigParser()
        self.name = name if name else default_config

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

    @classmethod
    @abstractmethod
    def _COMPONENT_TYPE(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining component type"""

    @classmethod
    @abstractmethod
    def _CONFIG_VERSION(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining component type"""

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

        return self._cfg.get(section, key, **kwargs)

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
            raise IOError(
                ErrorNumbers.FB600.value + ": cannot save config file: " + self.path
            ) from exp

    def generate(self, force: bool = False, id: Optional[str] = None) -> bool:
        """ "Generate configuration file

        Args:
            force: Overwrites existing configration file
            id: Component ID
        """

        # Check if configuration is already existing
        if self.is_config_existing() and not force:
            return self.read()

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
        return self.write()

    @abstractmethod
    def add_parameters(self):
        """ "Component specific argument creation"""

    def refresh(self):
        """Refreshes config file by recreating all the fields without
        chaning component ID.
        """

        if not self.is_config_existing():
            raise FedbiomedError("Can not refresh config file that is not existing")

        # Read the config
        self._cfg.read(self.path)
        id = self._cfg["default"]["id"]

        # Generate by keeping the component ID
        self.generate(force=True, id=id)
