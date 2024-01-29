# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os

from fedbiomed.common.constants import (
    SERVER_certificate_prefix,
    __researcher_config_version__,
    CONFIG_FOLDER_NAME
)

from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.config import Config


class ResearcherConfig(Config):

    _DEFAULT_CONFIG_FILE_NAME: str = 'config_researcher.ini'
    _COMPONENT_TYPE: str = 'RESEARCHER'
    _CONFIG_VERSION: str = __researcher_config_version__

    def add_parameters(self):
        """Generate researcher config"""

        grpc_host = os.getenv('RESEARCHER_SERVER_HOST', 'localhost')
        grpc_port = os.getenv('RESEARCHER_SERVER_PORT', '50051')

        # Generate certificate for gRPC server
        key_file, pem_file = generate_certificate(
            root=self.root,
            prefix=SERVER_certificate_prefix,
            component_id=self._cfg['default']['id'],
            subject={'CommonName': grpc_host}
        )

        self._cfg['server'] = {
            'host': grpc_host,
            'port': grpc_port,
            'pem' : os.path.relpath(pem_file, os.path.join(self.root, CONFIG_FOLDER_NAME)),
            'key' : os.path.relpath(key_file, os.path.join(self.root, CONFIG_FOLDER_NAME))
        }
