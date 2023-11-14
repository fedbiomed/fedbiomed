import os
import uuid

from fedbiomed.common.config import Config
from fedbiomed.common.utils import CONFIG_DIR
from fedbiomed.common.constants import \
    __researcher_config_version__ as __config_version__, \
    HashingAlgorithms


class NodeConfig(Config):

    DEFAULT_CONFIG_FILE_NAME: str = 'config_node.ini'
    COMPONENT_TYPE: str = 'NODE'
    CONFIG_VERSION: str = __config_version__

    def generate(self):
        """Generate researcher config"""

        node_id = os.getenv('NODE_ID', 'node_' + str(uuid.uuid4()))
        self._cfg['default'] = {'id': node_id}


        # Security variables
        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', True),
            'training_plan_approval': os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', False),
            'secure_aggregation': os.getenv('SECURE_AGGREGATION', True),
            'force_secure_aggregation': os.getenv('FORCE_SECURE_AGGREGATION', False)
        }

        # gRPC server host and port
        self._cfg["researcher"] = {
            'ip': os.getenv('RESEARCHER_SERVER_HOST', 'localhost'),
            'port': os.getenv('RESEARCHER_SERVER_PORT', '50051')
        }

        return super().generate()
