# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from fedbiomed.common.constants import DatasetTypes

from ._custom_dataset import CustomDataset
from ._dataset import Dataset
from ._medical_folder_dataset import MedicalFolderDataset
from ._native_dataset import NativeDataset
from ._simple_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)
from ._tabular_dataset import TabularDataset

DATASET_CLASSES_PER_TYPE = {
    DatasetTypes.CUSTOM: CustomDataset,
    DatasetTypes.IMAGES: ImageFolderDataset,
    DatasetTypes.MEDICAL_FOLDER: MedicalFolderDataset,
    DatasetTypes.MEDNIST: MedNistDataset,
    DatasetTypes.DEFAULT: MnistDataset,
    DatasetTypes.TABULAR: TabularDataset,
}

__all__ = [
    "Dataset",
    "CustomDataset",
    "ImageFolderDataset",
    "MedicalFolderDataset",
    "MedNistDataset",
    "MnistDataset",
    "NativeDataset",
    "TabularDataset",
]
