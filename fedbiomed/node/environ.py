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

Descriptions of the nodes Global Variables:

- NODE_ID                           : id of the node
- ID                                : equals to node id
- MESSAGES_QUEUE_DIR                : Path for queues
- DEFAULT_TRAINING_PLANS_DIR        : Path of directory for storing default training plans
- TRAINING_PLANS_DIR                 : Path of directory for storing registered training plans
- TRAINING_PLAN_APPROVAL            : True if the node enables training plan approval
- ALLOW_DEFAULT_TRAINING_PLANS      : True if the node enables default training plans for training plan approval
- HASHING_ALGORITHM                 : Hashing algorithm used for training plan approval
- SECURE_AGGREGATION                : True if secure aggregation is allowed on the node
- FORCE_SECURE_AGGREGATION          : True if secure aggregation is mandatory on the node
- EDITOR                            : Tool to use to edit training plan
- RESEARCHERS                       : List of researchers endpoint description

"""

import os
import sys

from fedbiomed.common.constants import (
    TRACEBACK_LIMIT,
    ComponentType,
    ErrorNumbers,
    HashingAlgorithms,
)
from fedbiomed.common.environ import Environ
from fedbiomed.common.exceptions import FedbiomedEnvironError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import ROOT_DIR
from fedbiomed.node.config import Config, node_component


class NodeEnviron(Environ):

    def __init__(self, root_dir: str = None, autoset: bool = True):
        """Constructs NodeEnviron object"""
        super().__init__()

        r = os.environ.get('FBM_NODE_COMPONENT_ROOT', root_dir)
        self._config: Config = node_component.create(r)
        self._root_dir = self._config.root

        loglevel = os.environ.get("FBM_LOG_LEVEL", "INFO")
        logger.setLevel(loglevel)

        self._values["COMPONENT_TYPE"] = ComponentType.NODE

        if autoset:
            self.set_environment()

    def set_environment(self):
        """Initializes environment variables"""

        # Sets common variable
        super().set_environment()

        node_id = self._config.get("default", "id")
        self._values["ID"] = node_id
        self._values["NODE_ID"] = node_id

        self._values["MESSAGES_QUEUE_DIR"] = os.path.join(
            self._values["VAR_DIR"], 'queue_manager'
        )

        self._values["DEFAULT_TRAINING_PLANS_DIR"] = os.path.join(
            ROOT_DIR, "envs", "common", "default_training_plans"
        )

        # default directory for saving training plans that are approved / waiting
        # for approval / rejected
        self._values["TRAINING_PLANS_DIR"] = os.path.join(
            self._values["VAR_DIR"], 'training_plans'
        )
        # Catch exceptions
        if not os.path.isdir(self._values["TRAINING_PLANS_DIR"]):
            # create training plan directory
            os.mkdir(self._values["TRAINING_PLANS_DIR"])

        allow_dtp = self._config.get("security", "allow_default_training_plans")

        self._values["ALLOW_DEFAULT_TRAINING_PLANS"] = os.getenv(
            "ALLOW_DEFAULT_TRAINING_PLANS", allow_dtp
        ).lower() in ("true", "1", "t", True)

        tp_approval = self._config.get("security", "training_plan_approval")

        self._values["TRAINING_PLAN_APPROVAL"] = os.getenv(
            "ENABLE_TRAINING_PLAN_APPROVAL", tp_approval
        ).lower() in ("true", "1", "t", True)

        hashing_algorithm = self._config.get("security", "hashing_algorithm")
        if hashing_algorithm in HashingAlgorithms.list():
            self._values["HASHING_ALGORITHM"] = hashing_algorithm
        else:
            raise FedbiomedEnvironError(
                f"{ErrorNumbers.FB600.value}: unknown hashing algorithm: {hashing_algorithm}"
            )

        secure_aggregation = self._config.get("security", "secure_aggregation")
        self._values["SECURE_AGGREGATION"] = os.getenv(
            "SECURE_AGGREGATION", secure_aggregation
        ).lower() in ("true", "1", "t", True)

        force_secure_aggregation = self._config.get(
            "security", "force_secure_aggregation"
        )
        self._values["FORCE_SECURE_AGGREGATION"] = os.getenv(
            "FORCE_SECURE_AGGREGATION", force_secure_aggregation
        ).lower() in ("true", "1", "t", True)

        self._values["EDITOR"] = os.getenv("EDITOR")

        public_key = self.config.get("certificate", "public_key")
        private_key = self.config.get("certificate", "private_key")
        self._values["FBM_CERTIFICATE_KEY"] = os.getenv(
            "FBM_CERTIFICATE_KEY", os.path.join(self._values["CONFIG_DIR"], private_key)
        )
        self._values["FBM_CERTIFICATE_PEM"] = os.getenv(
            "FBM_CERTIFICATE_PEM", os.path.join(self._values["CONFIG_DIR"], public_key)
        )

        # Parse each researcher ip and port
        researcher_sections = [
            section
            for section in self._config.sections()
            if section.startswith("researcher")
        ]

        self._values["RESEARCHERS"] = os.getenv("NODE_RESEARCHERS")
        if os.getenv("RESEARCHER_SERVER_HOST"):
            # Environ variables currently permit to specify only 1 researcher
            self._values["RESEARCHERS"] = [
                {
                    "port": os.getenv(
                        "RESEARCHER_SERVER_PORT", "50051"
                    ),  # use default port if not specified
                    "ip": os.getenv("RESEARCHER_SERVER_HOST"),
                    "certificate": None,
                }
            ]
        else:
            self._values["RESEARCHERS"] = []
            for section in researcher_sections:
                self._values["RESEARCHERS"].append(
                    {
                        "port": self._config.get(section, "port"),
                        "ip": self._config.get(section, "ip"),
                        "certificate": None,
                    }
                )

    def info(self):
        """Print useful information at environment creation"""

        logger.info(
            "type                           = " + str(self._values["COMPONENT_TYPE"])
        )
        logger.info(
            "training_plan_approval         = "
            + str(self._values["TRAINING_PLAN_APPROVAL"])
        )
        logger.info(
            "allow_default_training_plans   = "
            + str(self._values["ALLOW_DEFAULT_TRAINING_PLANS"])
        )


sys.tracebacklimit = TRACEBACK_LIMIT


# # global dictionary which contains all environment for the NODE
environ = NodeEnviron(autoset=True)
