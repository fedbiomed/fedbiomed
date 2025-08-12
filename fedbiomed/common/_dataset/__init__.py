# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from ._dataset import Dataset, StructuredDataset
from ._image_folder_dataset import ImageFolderDataset
from ._legacy_tabular_dataset import LegacyTabularDataset
from ._medical_datasets import (
    MedicalFolderBase,
    MedicalFolderController,
    MedicalFolderDataset,
    MedicalFolderLoadingBlockTypes,
)
from ._mnist_dataset import MnistDataset
from ._native_dataset import NativeDataset

__all__ = [
    "Dataset",
    "ImageFolderDataset",
    "LegacyTabularDataset",
    "MedicalFolderBase",
    "MedicalFolderController",
    "MedicalFolderDataset",
    "MedicalFolderLoadingBlockTypes",
    "NativeDataset",
    "MnistDataset",
    "StructuredDataset",
]
