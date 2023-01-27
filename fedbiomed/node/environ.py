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

import os
import uuid

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.constants import ComponentType, ErrorNumbers, HashingAlgorithms, DB_PREFIX, NODE_PREFIX
from fedbiomed.common.environ import Environ


class NodeEnviron(Environ):

    def __init__(self, root_dir: str = None):
        """Constructs NodeEnviron object """
        super().__init__(root_dir=root_dir)
        logger.setLevel("INFO")
        self._values["COMPONENT_TYPE"] = ComponentType.NODE
        # Setup environment variables
        self.setup_environment()

    def default_config_file(self) -> str:
        """Sets config file path """

        return os.path.join(self._values['CONFIG_DIR'], 'config_node.ini')

    def _set_component_specific_variables(self):
        """Initializes environment variables """

        node_id = self.from_config('default', 'id')
        self._values['NODE_ID'] = os.getenv('NODE_ID', node_id)
        self._values['ID'] = self._values['NODE_ID']

        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(self._values['VAR_DIR'],
                                                          f'queue_manager_{self._values["NODE_ID"]}')
        self._values['DB_PATH'] = os.path.join(self._values['VAR_DIR'],
                                               f'{DB_PREFIX}{self._values["NODE_ID"]}.json')

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

        allow_dtp = self.from_config('security', 'allow_default_training_plans')

        self._values['ALLOW_DEFAULT_TRAINING_PLANS'] = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', allow_dtp) \
            .lower() in ('true', '1', 't', True)

        tp_approval = self.from_config('security', 'training_plan_approval')

        self._values['TRAINING_PLAN_APPROVAL'] = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', tp_approval) \
            .lower() in ('true', '1', 't', True)

        hashing_algorithm = self.from_config('security', 'hashing_algorithm')
        if hashing_algorithm in HashingAlgorithms.list():
            self._values['HASHING_ALGORITHM'] = hashing_algorithm
        else:
            _msg = ErrorNumbers.FB600.value + ": unknown hashing algorithm: " + str(hashing_algorithm)
            logger.critical(_msg)
            raise FedbiomedEnvironError(_msg)

        self._values['EDITOR'] = os.getenv('EDITOR')

        # ========= PATCH MNIST Bug torchvision 0.9.0 ===================
        # https://github.com/pytorch/vision/issues/1938

        # imported only for the node component
        from six.moves import urllib

        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-agent', 'Python-urllib/3.7'),
            ('Accept',
             'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
            ('Accept-Language', 'en-US,en;q=0.9'),
            ('Accept-Encoding', 'gzip, deflate, br')
        ]
        urllib.request.install_opener(opener)

    def _set_component_specific_config_parameters(self):
        """Updates config file with Node specific parameters"""

        # TODO: We may remove node_id in the future (to simplify the code)
        node_id = os.getenv('NODE_ID', NODE_PREFIX + str(uuid.uuid4()))
        uploads_url = self._get_uploads_url()

        self._cfg['default'] = {
            'id': node_id,
            'component': "NODE",
            'uploads_url': uploads_url
        }

        # Security variables
        # Default hashing algorithm is SHA256
        allow_default_training_plans = os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', True)
        training_plan_approval = os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', False)

        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': allow_default_training_plans,
            'training_plan_approval': training_plan_approval
        }

    def info(self):
        """Print useful information at environment creation"""

        logger.info("type                           = " + str(self._values['COMPONENT_TYPE']))
        logger.info("training_plan_approval         = " + str(self._values['TRAINING_PLAN_APPROVAL']))
        logger.info("allow_default_training_plans   = " + str(self._values['ALLOW_DEFAULT_TRAINING_PLANS']))


# global dictionary which contains all environment for the NODE
environ = NodeEnviron()
