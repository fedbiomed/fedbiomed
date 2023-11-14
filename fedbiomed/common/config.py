import configparser
import os 
import uuid 

from abc import ABC

from fedbiomed.common.constants import ErrorNumbers, MPSPDZ_certificate_prefix, CONFIG_FOLDER_NAME
from fedbiomed.common.utils import  raise_for_version_compatibility, CONFIG_DIR, ROOT_DIR
from fedbiomed.common.certificate_manager import retrieve_ip_and_port, generate_certificate


class Config(ABC):
    """Base Config class"""

    def __init__(self, root = None):
        """Initializes config"""
        self._cfg = configparser.ConfigParser()

        config_file_name = os.getenv("CONFIG_FILE", self.DEFAULT_CONFIG_FILE_NAME)

        if root: 
            self.path = os.path.join(root, CONFIG_FOLDER_NAME, config_file_name)
            self.root = root
        else:
            self.path = os.path.join(CONFIG_DIR, config_file_name)
            self.root = ROOT_DIR

        if os.path.isfile(self.path):
            self._cfg.read(self.path)

            # Validate config version
            raise_for_version_compatibility(
                self._cfg["default"]["version"], 
                self.CONFIG_VERSION,
                f"Configuration file {self.path}: found version %s expected version %s")
        else:
            self.generate()

    def get(self, section, key) -> str:
        """Returns value for given ket and section"""

        return self._cfg.get(section, key)

    def sections(self) -> list:
        """Returns sections of the config"""

        return self._cfg.sections()

    def generate(self) -> bool:
        """"Generate configuration file"""

        self._cfg['default']['component'] = self.COMPONENT_TYPE 
        self._cfg['default']['version'] = str(self.CONFIG_VERSION)


        ip, port = retrieve_ip_and_port(self.root)
        component_id = f"component_{uuid.uuid4()}"
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
            'allow_default_biprimes': allow_default_biprimes
        }

        try:
            with open(self.path, 'w') as f:
                self._cfg.write(f)
        except configparser.Error:  
            raise IOError(ErrorNumbers.FB600.value + ": cannot save config file: " + self.path)

        return True
