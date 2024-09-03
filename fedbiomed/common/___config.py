# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
import uuid

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Dict

from fedbiomed.common.constants import (
    ErrorNumbers,
    MPSPDZ_certificate_prefix,
    CONFIG_FOLDER_NAME,
    VAR_FOLDER_NAME,
    TMP_FOLDER_NAME,
    CACHE_FOLDER_NAME,
    DB_PREFIX
)
from fedbiomed.common.utils import (
    create_fedbiomed_setup_folders,
    raise_for_version_compatibility,
    CONFIG_DIR,
    ROOT_DIR
)
from fedbiomed.common.certificate_manager import retrieve_ip_and_port, generate_certificate
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.secagg_manager import SecaggBiprimeManager


class Component:

    _COMPONENT_TYPE: str
    _CONFIG_CLASS: Config

    def __init__(
        self,
        path: str | None =  None
    ) -> None:

        # If there is path provided use it otherwise it is the working directory
        self._path = path if path else os.getcwd()
        self._create()
        self._config = _CONFIG_CLASS(
            root = path
        )

    def _create(self) -> None:
        """ Creates component folder and instantiate a config object"""

        if not os.path.isdir(self._path):
            os.makedirs(self._path)
        else:
            self._validate_root_path()


        var_dir = os.path.join(root, VAR_FOLDER_NAME)
        cache_dir = os.path.join(var_dir, CACHE_FOLDER_NAME)
        tmp_dir  = os.path.join(var_dir, TMP_FOLDER_NAME)

        for folder in [*self._FOLDERS, var_dir, cache_dir, tmp_dir]:
            pathlib.Path(os.path.join(self._path, folder)).mkdir(exist_ok=True)




    def _validate_root_path(self) -> None:
        """Validate if a new component can be created in given path"""

        fedbiomed_com = os.path.join(self._path, '.fedbiomed-component'),
        if os.path.isfile(fedbiomed_com):
            f = open(fedbiomed_com, 'r', encoding="UTF-8")
            component = f.read()

            if component != self._COMPONENT_TYPE:
                raise FedbiomedError(
                    f"There is a different component already instatiated in the given "
                    f"component root path {self._path}. Component already instaiated "
                    f"{component}, can not create component {self._COMPONENT} "
                )
        else:
            f = open(os.path.join(self._path, '.fedbiomed-component'), "a")
            f.write(f"{self._COMPONENT_TYPE}")
            f.close()



class Config(metaclass=ABCMeta):
    """Base Config class"""

    _CONFIG_VERSION: str

    def __init__(
        self,
        root: str,
        auto_generate: bool = True
    ) -> None:
        """Initializes config"""

        # First try to get component specific config file name, then CONFIG_FILE
        self._root = root
        self._cfg = configparser.ConfigParser()
        self._config_file = os.path.join(self._root, 'config.ini')

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

        return os.path.isfile(self._config_file)

    def read(self) -> bool:
        """Reads configuration file that is already existing in given path

        Raises verision compatibility error
        """

        self._cfg.read(self._config_file)

        # Validate config version
        raise_for_version_compatibility(
            self._cfg["default"]["version"],
            self._CONFIG_VERSION,
            f"Configuration file {self.path}: found version %s expected version %s")

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
            with open(self.path, 'w') as f:
                self._cfg.write(f)
        except configparser.Error:
            raise IOError(ErrorNumbers.FB600.value + ": cannot save config file: " + self.path)


    def _default(
        self,
        id: Optional[str] = None
    ) -> Dict:
        """Genereates default section of configuration

        Args:
          id: Component ID
        """


        return self._cfg['default']

    def generate(
        self,
        force: bool = False,
        id: Optional[str] = None
    ) -> bool:
        """"Generate configuration file

        Args:
            force: Overwrites existing configration file
            id: Component ID
        """

        # Check if configuration is already existing
        if self.is_config_existing() and not force:
            return self.read()


        # Create default section
        component_id = id if id else f"{self._COMPONENT_TYPE}_{uuid.uuid4()}"

        self._cfg['default'] = {
            'id': component_id,
            'component': self._COMPONENT_TYPE,
            'version': str(self._CONFIG_VERSION)
        }

        db_path  = os.path.join(self._root, f"{DB_PREFIX}{component_id}.json")
        self._cfg['default']['db'] = os.path.relpath(db_path, os.path.join(self.root, CONFIG_FOLDER_NAME))

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

        # Register default biprime
        self._register_default_biprime(db_path)

        # Calls child class add_parameterss
        self.add_parameters()

        # Write configuration file
        return self.write()

    @abstractmethod
    def add_parameters(self):
        """"Component specific argument creation"""

    def refresh(self):
        """Refreshes config file by recreating all the fields without
          chaning component ID.
        """

        if not self.is_config_existing():
            raise FedbiomedError("Can not refresh config file that is not existing")

        # Read the config
        self._cfg.read(self.path)
        id = self._cfg["default"]['id']

        # Generate by keeping the component ID
        self.generate(force=True, id=id)

    def _register_default_biprime(self, db_path: str):
        """Registers default biprime into database

        Args:
            db_path: The path to component's DB file.
        """

        df_biprimes = self._cfg.get('mpspdz', 'allow_default_biprimes')
        biprimes_dir = os.path.normpath(
            os.path.join(self.root, self.name, self._cfg.get('mpspdz', 'default_biprimes_dir'))
        )
        # Update secure aggregation biprimes in component database
        print(
            "Updating secure aggregation default biprimes with:\n"
            f"ALLOW_DEFAULT_BIPRIMES : {df_biprimes}\n"
            f"DEFAULT_BIPRIMES_DIR   : {biprimes_dir}\n"
        )

        BPrimeManager = SecaggBiprimeManager(db_path)
        BPrimeManager.update_default_biprimes(df_biprimes, biprimes_dir)

