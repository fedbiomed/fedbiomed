# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from enum import Enum

from fedbiomed.common.constants import DataLoadingBlockTypes

from ._dataset import Dataset
from ._simple_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)
from ._tabular_dataset import TabularDataset


# TODO - DATASET REDESIGN - remove when redesign is complete, and find place to put it
class MedicalFolderLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    MODALITIES_TO_FOLDERS: str = "modalities_to_folders"


__all__ = [
    "Dataset",
    "ImageFolderDataset",
    "MedNistDataset",
    "MnistDataset",
    "TabularDataset",
]
