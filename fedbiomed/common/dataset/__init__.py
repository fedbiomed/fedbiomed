# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from ._dataset import Dataset, NativeDataset, StructuredDataset
from ._legacy_tabular_dataset import LegacyTabularDataset
from ._medical_datasets import MedicalFolderDataset, MedicalFolderBase, MedicalFolderController, \
    MedicalFolderLoadingBlockTypes


__all__ = [
    "Dataset",
    "NativeDataset",
    "StructuredDataset",
    "LegacyTabularDataset",
    "MedicalFolderBase",
    "MedicalFolderController",
    "MedicalFolderDataset",
    "MedicalFolderLoadingBlockTypes",
]
