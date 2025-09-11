# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Generic data manager
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from fedbiomed.common.dataloader import DataLoader
from fedbiomed.common.dataset import Dataset


class FrameworkDataManager(ABC):
    """Class for creating data loaders from dataset depending on training plans"""

    _dataset: Dataset

    @abstractmethod
    def __init__(self, dataset: Dataset, **kwargs: dict):
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader
        """

    @property
    def dataset(self) -> Dataset:
        """Gets dataset.

        Returns:
            Dataset instance
        """
        return self._dataset

    @abstractmethod
    def split(
        self,
        test_ratio: float,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Split dataset and return data loaders"""

    @abstractmethod
    def save_state(self) -> Dict:
        """Gets state of the data loader.

        Returns:
            A Dict containing data loader state.
        """

    @abstractmethod
    def load_state(self, state: Dict):
        """Loads state of the data loader


        It currently keep only testing index, training index and test ratio
        as state.

        Args:
            state: Object containing data loader state.
        """
