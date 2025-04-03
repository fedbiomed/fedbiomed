# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import shutil
from typing import Optional

from fedbiomed.common.constants import (
    SERVER_certificate_prefix,
    __researcher_config_version__,
    CONFIG_FOLDER_NAME,
    VAR_FOLDER_NAME,
    TENSORBOARD_FOLDER_NAME,
    DEFAULT_RESEARCHER_NAME,
    NOTEBOOKS_FOLDER_NAME,
    TUTORIALS_FOLDER_NAME,
    DOCS_FOLDER_NAME,
)
from fedbiomed.common.utils import SHARE_DIR
from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.logger import logger
from fedbiomed.common.config import Component, Config


class ResearcherConfig(Config):

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
    "FBM_RESEARCHER_COMPONENT_ROOT", None
)


class ResearcherComponent(Component):
    """Fed-BioMed Node Component Class

    This class is used for creating and validating components
    by given component root directory
    """
    config_cls = ResearcherConfig
    _default_component_name = DEFAULT_RESEARCHER_NAME

    def initiate(self, root: Optional[str] = None) -> ResearcherConfig:
        """Creates or initiates existing component"""
        config = super().initiate(root)

        notebooks_path = os.path.join(config.root, NOTEBOOKS_FOLDER_NAME)
        notebooks_share_path = os.path.join(SHARE_DIR, NOTEBOOKS_FOLDER_NAME)
        docs_share_path = os.path.join(SHARE_DIR, DOCS_FOLDER_NAME)
        if not os.path.isdir(notebooks_path):
            shutil.copytree(notebooks_share_path, notebooks_path, symlinks=True)
            shutil.copytree(
                os.path.join(docs_share_path, TUTORIALS_FOLDER_NAME),
                os.path.join(notebooks_path, TUTORIALS_FOLDER_NAME),
                symlinks=True,
                dirs_exist_ok=True,
            )

        return config


researcher_component = ResearcherComponent()
config = researcher_component.initiate(root=component_root)
