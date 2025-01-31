# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

from fedbiomed.common.constants import (
    DEFAULT_CERT_NAME,
    HashingAlgorithms,
    __node_config_version__,
    DEFAULT_NODE_NAME,
    NODE_DATA_FOLDER
)
from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.config import Component, Config


class NodeConfig(Config):

    _CONFIG_VERSION: str = __node_config_version__
    COMPONENT_TYPE: str = 'NODE'

    def add_parameters(self):
        """Generate `Node` config"""

        # Security variables
        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': os.getenv('FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS', 'True'),
            'training_plan_approval': os.getenv('FBM_SECURITY_TRAINING_PLAN_APPROVAL', 'False'),
            'secure_aggregation': os.getenv('FBM_SECURITY_SECURE_AGGREGATION', 'True'),
            'force_secure_aggregation': os.getenv('FBM_SECURITY_FORCE_SECURE_AGGREGATION', 'False'),
            'secagg_insecure_validation': os.getenv('FBM_SECURITY_SECAGG_INSECURE_VALIDATION', 'True'),
        }
        # Generate self-signed certificates
        key_file, pem_file = generate_certificate(
            root=self.root, component_id=self._cfg["default"]["id"], prefix=DEFAULT_CERT_NAME
        )

        self._cfg["certificate"] = {
            "private_key": os.path.relpath(key_file, os.path.join(self.root, "etc")),
            "public_key": os.path.relpath(pem_file, os.path.join(self.root, "etc"))
        }

        # gRPC server host and port
        self._cfg["researcher"] = {
            'ip': os.getenv('FBM_RESEARCHER_IP', 'localhost'),
            'port': os.getenv('FBM_RESEARCHER_PORT', '50051')
        }


component_root = os.environ.get(
    "FBM_NODE_COMPONENT_ROOT", None
)


class NodeComponent(Component):
    """Fed-BioMed Node Component Class

    This class is used for creating and validating components
    by given component root directory
    """
    config_cls = NodeConfig
    _default_component_name = DEFAULT_NODE_NAME

    def initiate(self, root: Optional[str] = None) -> NodeConfig:
        config = super().initiate(root)
        node_data_path = os.path.join(config.root, NODE_DATA_FOLDER)
        os.makedirs(node_data_path, exist_ok=True)
        return config


node_component = NodeComponent()
