# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MedicalFolderDataset
from fedbiomed.common.logger import logger

logger.warning(
    "Importing data classes from the `fedbiomed.common.data` "
    "package has been deprecated. "
    "Please use the following import statements instead:\n"
    "`from fedbiomed.common.datamanager import DataManager`\n"
    "`from fedbiomed.common.dataset.flamby_dataset import FlambyDataset`\n"
    "`from fedbiomed.common.dataset import MedicalFolderDataset`"
)

__all__ = [
    "DataManager",
    "MedicalFolderDataset",
]
