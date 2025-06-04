# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataloader
"""

from ._np_dataloader import NPDataLoader
from ._dataloader import DataLoader
from ._pytorch_dataloader import PytorchDataLoader
from ._sklearn_dataloader import SkLearnDataLoader

__all__ = [
    "DataLoader",
    "PytorchDataLoader",
    "SkLearnDataLoader",
    "NPDataLoader",
]
