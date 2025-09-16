# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for scikit-learn training plan
"""

from typing import List, Optional, Tuple

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloader import SkLearnDataLoader
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError

from ._framework_data_manager import FrameworkDataManager, FrameworkSubset


class _SkLearnSubset(Dataset, FrameworkSubset):
    """Subset of a dataset at specified indices."""

    dataset: Dataset
    indices: List[int]

    def __init__(self, dataset: Dataset, indices: List[int]) -> None:
        """Class constructor

        Args:
            dataset: Dataset from which to take the subset.
            indices: Indices in the whole set selected for subset.
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        """Gets the length of the subset.

        Returns:
            Length of the subset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Gets an item from the subset.

        Args:
            idx: Index of the item in the subset.

        Raises:
            FedbiomedError: if `idx` is out of range

        Returns:
            Item at index `idx` in the subset.
        """
        if idx < 0 or idx >= len(self):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}. Bad index {idx} in subset of length {len(self)}."
            )

        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def complete_initialization(self) -> None:
        """Unused method to comply with Dataset interface."""
        pass


class SkLearnDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for scikit-learn training plans"""

    _loader_class = SkLearnDataLoader
    _subset_class = _SkLearnSubset
    _dataset_wrapper = lambda cls, x: x  # noqa: E731 - avoid declaring a dummy function

    _subset_test: Optional[_SkLearnSubset] = None
    _subset_train: Optional[_SkLearnSubset] = None

    def __init__(self, dataset: Dataset, **kwargs: dict):
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader
        """
        super().__init__(dataset, **kwargs)

        # Manage randomization for SkLearnDataManager and SkLearnDataLoader through np.random()
        # Control seed here to manage reproducibility for both classes
        #
        # TODO: random_seed / random_state should be standardized across all DataManagers
        seed = self._loader_arguments.pop("random_seed", None)
        if isinstance(seed, int):
            np.random.seed(seed)
        else:
            # reset seed to a random value to ensure non-deterministic behavior
            # if no seed is specified
            np.random.seed()

        self._dataset.to_format = DataReturnFormat.SKLEARN

    # Nota: used only for unit tests
    def subset_test(self) -> Optional[_SkLearnSubset]:
        """Gets validation subset of the dataset.

        Returns:
            Validation subset
        """

        return self._subset_test

    # Nota: used only for unit tests
    def subset_train(self) -> Optional[_SkLearnSubset]:
        """Gets train subset of the dataset.

        Returns:
            Train subset
        """
        return self._subset_train

    def _random_split(
        self, dataset: Dataset, lengths: List[int]
    ) -> Tuple[_SkLearnSubset, _SkLearnSubset]:
        """Randomly split a dataset into 2 non-overlapping subsets of given lengths.

        Args:
            dataset: Dataset to split
            lengths: Lengths of 2 splits to be produced

        Returns:
            List of subsets of the dataset. `None` if the subset is empty.
        """
        # No need to check types and lengths, already done in split()

        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        return (
            _SkLearnSubset(dataset, list(indices[: lengths[0]])),
            _SkLearnSubset(dataset, list(indices[lengths[0] :])),
        )
