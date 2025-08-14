# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

from ._dataset import DataReturnFormat, Dataset
from ._simple_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)

__all__ = [
    "Dataset",
    "DataReturnFormat",
    "ImageFolderDataset",
    "MedNistDataset",
    "MnistDataset",
]
