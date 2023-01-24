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

import os
import uuid

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.constants import ComponentType, ErrorNumbers, DB_PREFIX
from fedbiomed.common.environ import Environ


class ResearcherEnviron(Environ):

    def __init__(self, root_dir: str = None):
        """Constructs ResearcherEnviron object """
        super().__init__(root_dir=root_dir)
        logger.setLevel("DEBUG")
        self._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER

        # Setup environment variables
        self.setup_environment()

    def default_config_file(self) -> str:
        """Sets config file path """

        return os.path.join(self._values['CONFIG_DIR'], 'config_researcher.ini')

    def _set_component_specific_variables(self):

        # we may remove RESEARCHER_ID in the future (to simplify the code)
        # and use ID instead

        researcher_id = self.from_config('default', 'id')

        self._values['RESEARCHER_ID'] = os.getenv('RESEARCHER_ID', researcher_id)
        self._values['ID'] = self._values['RESEARCHER_ID']

        # more directories
        self._values['TENSORBOARD_RESULTS_DIR'] = os.path.join(self._values['ROOT_DIR'], 'runs')
        self._values['EXPERIMENTS_DIR'] = os.path.join(self._values['VAR_DIR'], "experiments")
        self._values['MESSAGES_QUEUE_DIR'] = os.path.join(self._values['VAR_DIR'], 'queue_messages')
        self._values['DB_PATH'] = os.path.join(self._values['VAR_DIR'],
                                               f'{DB_PREFIX}{self._values["RESEARCHER_ID"]}.json')
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

    def _set_component_specific_config_parameters(self):
        # get uploads url
        uploads_url = self._get_uploads_url()

        # Default configuration
        researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))
        self._cfg['default'] = {
            'id': researcher_id,
            'component': "RESEARCHER",
            'uploads_url': uploads_url
        }

    def info(self):
        """Print useful information at environment creation"""

        logger.info("Component environment:")
        logger.info("type = " + str(self._values['COMPONENT_TYPE']))


# Global dictionary which contains all environment for the RESEARCHER
environ = ResearcherEnviron()
