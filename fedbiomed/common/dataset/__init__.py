# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from ._dataset import Dataset, StructuredDataset
from ._native_dataset import NativeDataset
from ._legacy_tabular_dataset import LegacyTabularDataset
from ._medical_datasets import (
    MedicalFolderBase,
    MedicalFolderController,
    MedicalFolderDataset,
    MedicalFolderLoadingBlockTypes,
)

__all__ = [
    "Dataset",
    "StructuredDataset",
    "NativeDataset",
    "LegacyTabularDataset",
    "MedicalFolderBase",
    "MedicalFolderController",
    "MedicalFolderDataset",
    "MedicalFolderLoadingBlockTypes",
]
