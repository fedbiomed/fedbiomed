# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataloader
"""

from ._dataloader import DataLoader
from ._pytorch_dataloader import (
    PytorchDataLoader,
    PytorchDataLoaderItem,
    PytorchDataLoaderSample,
)
from ._sklearn_dataloader import (
    SkLearnDataLoader,
    SkLearnDataLoaderItemBatch,
    SkLearnDataLoaderSampleBatch,
)

__all__ = [
    "DataLoader",
    "PytorchDataLoaderItem",
    "PytorchDataLoaderSample",
    "PytorchDataLoader",
    "SkLearnDataLoaderItemBatch",
    "SkLearnDataLoaderSampleBatch",
    "SkLearnDataLoader",
]
