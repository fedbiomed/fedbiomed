import os
import uuid

from fedbiomed.common.config import Config
from fedbiomed.common.constants import SERVER_certificate_prefix, \
    __researcher_config_version__ as __config_version__, \
    ETC_FOLDER_NAME

from fedbiomed.common.certificate_manager import generate_certificate


class ResearcherConfig(Config):

    DEFAULT_CONFIG_FILE_NAME: str = 'config_researcher.ini'
    COMPONENT_TYPE: str = 'RESEARCHER'
    CONFIG_VERSION: str = __config_version__

    def generate(self):
        """Generate researcher config"""

        researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))
        self._cfg['default'] = { 'id': researcher_id }

        # Generate certificate for gRPC server
        key_file, pem_file = generate_certificate(
            root=self.root, 
            component_id=researcher_id, 
            prefix=SERVER_certificate_prefix)


        self._cfg['server'] = {
            'host': os.getenv('RESEARCHER_SERVER_HOST', 'localhost'),
            'port': os.getenv('RESEARCHER_SERVER_PORT', '50051'),
            'pem' : os.path.relpath(pem_file, os.path.join(self.root, ETC_FOLDER_NAME)),
            'key' : os.path.relpath(key_file, os.path.join(self.root, ETC_FOLDER_NAME))
        }


        return super().generate()
