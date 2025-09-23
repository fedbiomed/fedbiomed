# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract class for data loaders specific to a training plan's framework
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class DataLoader(ABC):
    """Abstract base class for data loaders specific to a training plan's framework"""

    @abstractmethod
    def __init__(self, dataset: Any, *args, **kwargs) -> None:
        """Class constructor"""

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of batches of the encapsulated dataset"""

    @abstractmethod
    def __iter__(self) -> Iterable:
        """Returns an iterator over batches of data"""

    # Implemented by public variables (torch) of properties (sklearn)
    #
    # @property
    # @abstractmethod
    # def dataset(self) -> Any:
    #     """Returns the encapsulated dataset"""
    #
    # @property
    # @abstractmethod
    # def batch_size(self) -> int:
    #     """Returns the batch size used by the data loader"""
    #
    # @property
    # @abstractmethod
    # def drop_last(self) -> bool:
    #     """Returns whether the data loader drops the last incomplete batch"""
