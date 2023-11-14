# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Module that initialize singleton environ object for the researcher component

[`Environ`][fedbiomed.common.environ] will be initialized after the object `environ`
is imported from `fedbiomed.researcher.environ`

**Typical use:**

```python
from fedbiomed.researcher.environ import environ

print(environ['RESEARCHER_ID'])
```
"""

import sys
import os
import uuid

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.constants import ComponentType, ErrorNumbers, DB_PREFIX, \
    TENSORBOARD_FOLDER_NAME
from fedbiomed.common.constants import __researcher_config_version__ as __config_version__
from fedbiomed.common.environ import Environ
from fedbiomed.researcher.config import ResearcherConfig


class ResearcherEnviron(Environ):

    def __init__(self, root_dir: str = None):
        """Constructs ResearcherEnviron object """
        super().__init__(root_dir=root_dir)

        self.config = ResearcherConfig(root_dir)

        logger.setLevel("DEBUG")
        # Set component type
        self._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER
        # Setup environment variables
        self.set()

    def set(self):

        super().set()

        # we may remove RESEARCHER_ID in the future (to simplify the code)
        # and use ID instead
        researcher_id = self.config.get('default', 'id')

        self._values['RESEARCHER_ID'] = os.getenv('RESEARCHER_ID', researcher_id)
        self._values['ID'] = self._values['RESEARCHER_ID']

        # more directories
        self._values['TENSORBOARD_RESULTS_DIR'] = os.path.join(self._values['ROOT_DIR'], TENSORBOARD_FOLDER_NAME)
        self._values['EXPERIMENTS_DIR'] = os.path.join(self._values['VAR_DIR'], "experiments")
        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(self._values['VAR_DIR'], 'queue_messages')
        self._values['DB_PATH'] = os.path.join(self._values['VAR_DIR'],
                                               f'{DB_PREFIX}{self._values["RESEARCHER_ID"]}.json')

        self._values["SERVER_HOST"] = os.getenv('RESEARCHER_SERVER_HOST', 
                                                self.config.get('server', 'host'))
        self._values["SERVER_PORT"] = os.getenv('RESEARCHER_SERVER_PORT', 
                                                self.config.get('server', 'port'))

        for _key in 'TENSORBOARD_RESULTS_DIR', 'EXPERIMENTS_DIR':
            dir = self._values[_key]
            if not os.path.isdir(dir):
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    _msg = ErrorNumbers.FB600.value + ": path already exists but is not a directory " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)
                except OSError:
                    _msg = ErrorNumbers.FB600.value + ": cannot create environment subtree in: " + dir
                    logger.critical(_msg)
                    raise FedbiomedEnvironError(_msg)

    def info(self):
        """Print useful information at environment creation"""

        logger.info("Component environment:")
        logger.info("type = " + str(self._values['COMPONENT_TYPE']))


sys.tracebacklimit = 3

# Global dictionary which contains all environment for the RESEARCHER
environ = ResearcherEnviron()
