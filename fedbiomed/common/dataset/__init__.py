# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from ._dataset import Dataset
from ._medical_folder_dataset import MedicalFolderDataset
from ._simple_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)

__all__ = [
    "Dataset",
    "ImageFolderDataset",
    "MedicalFolderDataset",
    "MedNistDataset",
    "MnistDataset",
]
