# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
import uuid

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Dict, Union
from packaging.version import Version

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
    read_file
)
from fedbiomed.common.exceptions import FedbiomedConfigurationError



def docker_special_case(component_path: str) -> bool:
    """Special case for docker containers.

    This function makes sure that there is only .gitkeep file present in
    the directory that component will be initialized. It is required since
    component folder should be existing in run_mounts by default.
    """

    files = os.listdir(component_path)

    return ".gitkeep" in files and len(files) == 1



class Config(metaclass=ABCMeta):
    """Base Config class

    Attributes:
        root: Root directory of the component.
        name: Config name (e.g config.ini or config-n1.ini).
        path: Absolute path to configuration.
        vars: A dictionary that contains configuration related variables. Such
            as dynamic paths that relies of component root etc.
    """

    _CONFIG_FILE_NAME: str = "config.ini"
    _CONFIG_VERSION: Version
    COMPONENT_TYPE: str

    _cfg: configparser.ConfigParser
    root: str
    path: str
    name: str

    vars: Dict[str, Any] = {}

    def __init__(
        self, root: str
    ) -> None:
        """Initializes configuration

        Args:
            root: Root directory for the component
        """
        self._cfg = configparser.ConfigParser()
        self.load(root)

    @classmethod
    @abstractmethod
    def COMPONENT_TYPE(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining component type"""

    @classmethod
    @abstractmethod
    def _CONFIG_VERSION(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining configuration version"""

    def load(
        self,
        root: str,
    ) -> None:
        """Load configuration from given name and root

        This implementation allows to load configuration after Config class
        is instantiated.

        Args:
            root: Root directory where component files will be saved
                configuration file.
        """

        self.root = root
        self.config_path = os.path.join(self.root, 'etc', self._CONFIG_FILE_NAME)
        self.generate()

    def is_config_existing(self) -> bool:
        """Checks if config file exists

        Returns:
            True if config file is already existing
        """

        return os.path.isfile(self.config_path)

    def read(self) -> bool:
        """Reads configuration file that is already existing in given path

        Raises version compatibility error
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
            with open(self.config_path, "w", encoding="UTF-8") as f:
                self._cfg.write(f)
        except configparser.Error as exp:
            raise FedbiomedConfigurationError(
                f"{ErrorNumbers.FB600.value}: cannot save config file:  {self.path}"
            ) from exp

    def generate(
        self,
        id: Optional[str] = None
    ) -> None:
        """ "Generate configuration file

        Args:
            force: Overwrites existing configration file
            id: Component ID
        """

        # Check if configuration is already existing
        if not self.is_config_existing():
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


class Component:

    config_cls: type
    _config: Config
    _default_component_name: str

    def __init__(self):
        """Test"""
        self._reference = '.fedbiomed'

    def initiate(self, root: Optional[str] = None) -> Union["NodeConfig", "ResearcherConfig"] :
        """Creates or initiates existing component"""

        if not root:
            root = os.path.join(os.getcwd(), self._default_component_name)

        reference = self.validate(root)
        config = self.config_cls(root)

        if not os.path.isfile(reference):
            create_fedbiomed_setup_folders(root)
            with open(os.path.join(root, '.fedbiomed'), 'w', encoding='UTF-8') as file_:
                file_.write(self.config_cls.COMPONENT_TYPE)
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
            if docker_special_case(component_dir):
                return False

            if os.listdir(component_dir) and not os.path.isfile(ref):
                raise ValueError(
                    f"Cannot create component. Path {component_dir} "
                    "is not empty for Fed-BioMed component initialization. Please "
                    f"remove folder {component_dir} or specify another path"
                )

        # Special case for docker container mounted folders
        # empty .fedbiomed is required to keep it
        if os.path.isfile(ref) and not read_file(ref):
            return False

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
            if comp_type != self.config_cls.COMPONENT_TYPE:
                raise ValueError(
                    f'Component directory has already been initilazed for component type {comp_type}'
                    ' can not overwrite or reuse it for component type '
                    f'{self.config_cls.COMPONENT_TYPE}')

        return ref

