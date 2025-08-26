# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataloadingplan
"""

from enum import Enum

from fedbiomed.common.constants import DataLoadingBlockTypes

from ._data_loading_plan import (
    DataLoadingBlock,
    DataLoadingPlan,
    DataLoadingPlanMixin,
    MapperBlock,
    SerializationValidation,  # keep it for documentation
)


# TODO - DATASET-REDESIGN: This loading block is removed and need to be added somewhere in medcial folder dataset definition
class MedicalFolderLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    MODALITIES_TO_FOLDERS: str = "modalities_to_folders"


__all__ = [
    "DataLoadingBlock",
    "MapperBlock",
    "DataLoadingPlan",
    "DataLoadingPlanMixin",
    "SerializationValidation",
]
