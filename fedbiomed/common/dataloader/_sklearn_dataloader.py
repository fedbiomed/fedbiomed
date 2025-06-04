# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Class for data loader in PyTorch training plans
"""

from typing import Optional
from collections.abc import Iterable

import numpy as np

from ._dataloader import DataLoader


# Can either instantiate, derive or re-implement `NPDataLoader`

# Type `np.ndarray` changes if we don't load all dataset (eg a custom `SkLearnDataset` ?)

# Iterable to be replaced by the class used, eg `_BatchIterator` currently

class SkLearnDataLoader(DataLoader):
    def __init__(
            self,
            dataset: np.ndarray,
            target: Optional[np.ndarray] = None,
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""

    @property
    def dataset(self) -> np.ndarray:
        """Returns the encapsulated dataset"""

    def __len__(self) -> int:
        """Returns the length of the encapsulated dataset"""

    def __iter__(self) -> Iterable:
        """Returns an iterator over batches of data"""
