# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os

from fedbiomed.common.constants import (
    DEFAULT_CERT_NAME,
    HashingAlgorithms,
    __node_config_version__
)
from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.config import Config


class NodeConfig(Config):

    _DEFAULT_CONFIG_FILE_NAME: str = 'config_node.ini'
    _COMPONENT_TYPE: str = 'NODE'
    _CONFIG_VERSION: str = __node_config_version__

    def add_parameters(self):
        """Generate `Node` config"""

        # Security variables
        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', 'True'),
            'training_plan_approval': os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', 'False'),
            'secure_aggregation': os.getenv('SECURE_AGGREGATION', 'True'),
            'force_secure_aggregation': os.getenv('FORCE_SECURE_AGGREGATION', 'False'),
            'secagg_insecure_validation': os.getenv('SECAGG_INSECURE_VALIDATION', 'True'),
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
            'ip': os.getenv('RESEARCHER_SERVER_HOST', 'localhost'),
            'port': os.getenv('RESEARCHER_SERVER_PORT', '50051')
        }
