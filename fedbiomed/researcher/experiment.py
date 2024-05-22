# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from .federated_workflows import Experiment
from fedbiomed.common.logger import logger

logger.warning("Importing Experiment class from the researcher package has been deprecated. "
               "Please use the following import statement instead:\n"
               "from fedbiomed.researcher.federated_workflows import Experiment")
