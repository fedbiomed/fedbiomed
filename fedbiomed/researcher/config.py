# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os

from fedbiomed.common.constants import (
    SERVER_certificate_prefix,
    __researcher_config_version__,
    CONFIG_FOLDER_NAME,
    VAR_FOLDER_NAME,
    TENSORBOARD_FOLDER_NAME,
    DEFAULT_CONFIG_FILE_NAME_RESEARCHER
)

from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.logger import logger
from fedbiomed.common.config import Component, Config


class ResearcherConfig(Config):

    _DEFAULT_CONFIG_FILE_NAME: str = DEFAULT_CONFIG_FILE_NAME_RESEARCHER
    _CONFIG_VERSION: str = __researcher_config_version__
    COMPONENT_TYPE: str = 'RESEARCHER'

    def add_parameters(self):
        """Generate researcher config"""

        grpc_host = os.getenv('FBM_SERVER_HOST', 'localhost')
        grpc_port = os.getenv('FBM_SERVER_PORT', '50051')

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
            'secagg_insecure_validation': os.getenv('FBM_SECURITY_SECAGG_INSECURE_VALIDATION', True)
        }


    def _update_vars(self):
        """Updates component dynamic vars"""

        super()._update_vars()

        self.vars.update({
            'EXPERIMENTS_DIR': os.path.join(self.root, VAR_FOLDER_NAME, 'experiments'),
            'TENSORBOARD_RESULTS_DIR': os.path.join(self.root, TENSORBOARD_FOLDER_NAME),
            'DB': os.path.join(self.root, CONFIG_FOLDER_NAME, self._cfg.get('default', 'db'))
        })

        os.makedirs(self.vars['EXPERIMENTS_DIR'], exist_ok=True)
        os.makedirs(self.vars['TENSORBOARD_RESULTS_DIR'], exist_ok=True)


logger.setLevel("DEBUG")

component_root = os.environ.get(
    "FBM_RESEARCHER_COMPONENT_ROOT", DEFAULT_CONFIG_FILE_NAME_RESEARCHER
)


class ResearcherComponent(Component):
    """Fed-BioMed Node Component Class

    This class is used for creating and validating components
    by given component root directory
    """
    _config_cls = ResearcherConfig
    _default_component_name = 'fbm-researcher'

researcher_component = ResearcherComponent()
config = researcher_component.create(root=component_root)
