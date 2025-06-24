# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataloader
"""

from ._dataloader import DataLoader
from ._np_dataloader import NPDataLoader
from ._pytorch_dataloader import (
    PytorchDataLoader,
    PytorchDataLoaderItem,
    PytorchDataLoaderSample,
)
from ._sklearn_dataloader import (
    SkLearnDataLoader,
    SkLearnDataLoaderItem,
    SkLearnDataLoaderSample,
)

__all__ = [
    "DataLoader",
    "PytorchDataLoaderItem",
    "PytorchDataLoaderSample",
    "PytorchDataLoader",
    "SkLearnDataLoaderItem",
    "SkLearnDataLoaderSample",
    "SkLearnDataLoader",
    "NPDataLoader",
]
