# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os

from fedbiomed.common.constants import (
    SERVER_certificate_prefix,
    __researcher_config_version__,
    CONFIG_FOLDER_NAME
)

from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.config import Component, Config


class ResearcherConfig(Config):

    _CONFIG_VERSION = __researcher_config_version__

    COMPONENT_TYPE: str = 'RESEARCHER'

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
        }

        self._cfg["certificate"] = {
            "private_key": os.path.relpath(key_file, os.path.join(self.root, CONFIG_FOLDER_NAME)),
            "public_key": os.path.relpath(pem_file, os.path.join(self.root, CONFIG_FOLDER_NAME))
        }

        self._cfg['security'] = {
            'secagg_insecure_validation': os.getenv('SECAGG_INSECURE_VALIDATION', True)
        }


class ResearcherComponent(Component):
    """Fed-BioMed Node Component Class

    This class is used for creating and validating components
    by given component root directory
    """
    config_cls = ResearcherConfig
    _default_component_name = 'fbm-researcher'

researcher_component = ResearcherComponent()
