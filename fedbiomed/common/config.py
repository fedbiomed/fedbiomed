# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
import uuid

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Dict
from packaging.version import Version

from fedbiomed.common.constants import (
    ErrorNumbers,
    CONFIG_FOLDER_NAME,
    VAR_FOLDER_NAME,
    DB_PREFIX,
)
from fedbiomed.common.utils import (
    create_fedbiomed_setup_folders,
    raise_for_version_compatibility,
    read_file
)
from fedbiomed.common.exceptions import FedbiomedError


class Config(metaclass=ABCMeta):
    """Base Config class"""

    _CONFIG_FILE_NAME: str = "config.ini"
    _CONFIG_VERSION: Version

    COMPONENT_TYPE: str

    def __init__(
        self, root: str, auto_generate: bool = True
    ) -> None:
        """Initializes config"""

        self.root = root
        self.config_path = os.path.join(self.root, 'etc', self._CONFIG_FILE_NAME)

        self._cfg = configparser.ConfigParser()

        if auto_generate:
            self.generate()

    def is_config_existing(self) -> bool:
        """Checks if config file exists

        Returns:
            True if config file is already existing
        """

        return os.path.isfile(self.config_path)

    def read(self) -> bool:
        """Reads configuration file that is already existing in given path

        Raises verision compatibility error
        """

        self._cfg.read(self.config_path)

        # Validate config version
        raise_for_version_compatibility(
            self._cfg["default"]["version"],
            self._CONFIG_VERSION,
            f"Configuration file {self.config_path}: found version %s expected version %s",
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
            with open(self.config_path, "w", encoding="UTF-8") as f:
                self._cfg.write(f)
        except configparser.Error as exp:
            raise IOError(
                ErrorNumbers.FB600.value + ": cannot save config file: " + self.config_path
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
        component_id = id if id else f"{self.COMPONENT_TYPE}_{uuid.uuid4()}"

        self._cfg["default"] = {
            "id": component_id,
            "component": self.COMPONENT_TYPE,
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
        return True
        # return self.write()

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


class Component:

    _config_cls: type
    _config: Config

    def __init__(self):
        """Test"""
        self._reference = '.fedbiomed'

    def create(self, root: str | None = None) -> Config:
        """Create component"""

        if not root:
            root = os.path.join(os.getcwd(), self._default_component_name)

        reference = self.validate(root)
        config = self._config_cls(root, auto_generate=False)

        if not os.path.isfile(reference):
            create_fedbiomed_setup_folders(root)
            with open(os.path.join(root, '.fedbiomed'), 'w', encoding='UTF-8') as file_:
                file_.write(self._config_cls.COMPONENT_TYPE)
            config.generate()
            config.write()
        else:
            config.read()

        return config


    def is_component_existing(self, component_dir: str) -> bool:
        """Checks if component existing in the given root directory

        Returns:
            True if any component is instantiated in the given directory
        """
        ref = os.path.join(component_dir, self._reference)
        if os.path.isdir(component_dir):
            if os.listdir(component_dir) and not os.path.isfile(ref):
                raise ValueError(
                    f'Path {component_dir} is not empty for Fed-BioMed component initialization.'
                )
        return os.path.isfile(ref)

    def validate(self, root) -> str:
        """Validates given root folder is a component can be instantiated

        Args:
            root: Root directory that Fed-BioMed component will be instantiated.

        Returns:
            Full path to reference file
        """

        iscomp = self.is_component_existing(root)
        ref = os.path.join(root, self._reference)

        if iscomp:
            comp_type = read_file(ref)
            if comp_type != self._config_cls.COMPONENT_TYPE:
                raise ValueError(
                    f'Component directory has already been initilazed for component type {comp_type}'
                    ' can not overwrite or reuse it for component type '
                    f'{self._config_cls.COMPONENT_TYPE}')

        return ref
