# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract class for data loaders specific to a training plan's framework
"""

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Iterable


class DataLoader(ABC):
    @abstractmethod
    def __init__(self, dataset: Any, *args, **kwargs) -> None:
        """Class constructor"""

    @property
    @abstractmethod
    def dataset(self) -> Any:
        """Returns the encapsulated dataset"""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the encapsulated dataset"""

    @abstractmethod
    def __iter__(self) -> Iterable:
        """Returns an iterator over batches of data"""