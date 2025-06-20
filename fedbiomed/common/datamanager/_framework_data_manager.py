# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Generic data manager
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

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

    @abstractmethod
    def split(
        self,
        test_ratio: float,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Split dataset and return data loaders"""

    # Maybe can factor some other class ?
