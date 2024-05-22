# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Module that initialize singleton environ object for the node component

[`Environ`][fedbiomed.common.environ] will be initialized after the object `environ`
is imported from `fedbiomed.node.environ`

**Typical use:**

```python
from fedbiomed.node.environ import environ

print(environ['NODE_ID'])
```

"""

import sys
import os

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.constants import (
    ComponentType,
    ErrorNumbers,
    HashingAlgorithms,
    TRACEBACK_LIMIT)
from fedbiomed.common.environ import Environ
from fedbiomed.node.config import NodeConfig


class NodeEnviron(Environ):

    def __init__(
        self,
        root_dir: str = None,
        autoset: bool = True
    ):
        """Constructs NodeEnviron object """
        super().__init__(root_dir=root_dir)

        self._root_dir = root_dir

        logger.setLevel("INFO")
        self._values["COMPONENT_TYPE"] = ComponentType.NODE

        if autoset:
            self.set_environment()


    def set_environment(self):
        """Initializes environment variables """

        self._config = NodeConfig(self._root_dir)

        # Sets common variable
        super().set_environment()

        node_id = self._config.get('default', 'id')
        self._values['ID'] = node_id
        self._values['NODE_ID'] = node_id

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(self._values['VAR_DIR'],
                                                          f'queue_manager_{self._values["NODE_ID"]}')

        self._values['DEFAULT_TRAINING_PLANS_DIR'] = os.path.join(self._values['ROOT_DIR'],
                                                                  'envs', 'common', 'default_training_plans')

        # default directory for saving training plans that are approved / waiting for approval / rejected
        self._values['TRAINING_PLANS_DIR'] = os.path.join(self._values['VAR_DIR'],
                                                          f'training_plans_{self._values["NODE_ID"]}')
        # FIXME: we may want to change that
        # Catch exceptions
        if not os.path.isdir(self._values['TRAINING_PLANS_DIR']):
            # create training plan directory
            os.mkdir(self._values['TRAINING_PLANS_DIR'])

        allow_dtp = self._config.get('security', 'allow_default_training_plans')

        self._values['ALLOW_DEFAULT_TRAINING_PLANS'] = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', allow_dtp) \
            .lower() in ('true', '1', 't', True)

        tp_approval = self._config.get('security', 'training_plan_approval')

        self._values['TRAINING_PLAN_APPROVAL'] = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', tp_approval) \
            .lower() in ('true', '1', 't', True)

        hashing_algorithm = self._config.get('security', 'hashing_algorithm')
        if hashing_algorithm in HashingAlgorithms.list():
            self._values['HASHING_ALGORITHM'] = hashing_algorithm
        else:
            _msg = ErrorNumbers.FB600.value + ": unknown hashing algorithm: " + str(hashing_algorithm)
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        secure_aggregation = self._config.get('security', 'secure_aggregation')
        self._values["SECURE_AGGREGATION"] = os.getenv('SECURE_AGGREGATION',
                                                       secure_aggregation).lower() in ('true', '1', 't', True)

        force_secure_aggregation = self._config.get('security', 'force_secure_aggregation')
        self._values["FORCE_SECURE_AGGREGATION"] = os.getenv(
            'FORCE_SECURE_AGGREGATION',
            force_secure_aggregation).lower() in ('true', '1', 't', True)

        self._values['EDITOR'] = os.getenv('EDITOR')


        # Parse each researcher ip and port
        researcher_sections = [section for section in self._config.sections() if section.startswith("researcher")]
        self._values['RESEARCHERS'] = os.getenv('NODE_RESEARCHERS')
        if os.getenv('RESEARCHER_SERVER_HOST'):
            # Environ variables currently permit to specify only 1 researcher
            self._values["RESEARCHERS"] = [
                {
                    'port': os.getenv('RESEARCHER_SERVER_PORT', '50051'),  # use default port if not specified
                    'ip': os.getenv('RESEARCHER_SERVER_HOST'),
                    'certificate': None
                }
            ]
        else:
            self._values["RESEARCHERS"] = []
            for section in researcher_sections:
                self._values["RESEARCHERS"].append({
                    'port': self._config.get(section, "port"),
                    'ip': self._config.get(section, "ip"),
                    'certificate': None
                })

    def info(self):
        """Print useful information at environment creation"""

        logger.info("type                           = " + str(self._values['COMPONENT_TYPE']))
        logger.info("training_plan_approval         = " + str(self._values['TRAINING_PLAN_APPROVAL']))
        logger.info("allow_default_training_plans   = " + str(self._values['ALLOW_DEFAULT_TRAINING_PLANS']))


sys.tracebacklimit = TRACEBACK_LIMIT


# # global dictionary which contains all environment for the NODE
environ = NodeEnviron(autoset=True)
