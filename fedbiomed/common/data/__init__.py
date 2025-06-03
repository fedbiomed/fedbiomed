# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.logger import logger

logger.warning("Importing `DataManager` class from the `fedbiomed.common.data` "
               "package has been deprecated. "
               "Please use the following import statement instead:\n"
               "`from fedbiomed.common.datamanager import DataManager`")

__all__ = [
    "DataManager",
]
