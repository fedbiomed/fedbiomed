# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Generic data manager
"""

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloader import DataLoader
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


class FrameworkSubset(ABC):
    """Subset of a dataset at specified indices."""

    dataset: Dataset
    indices: List[int]

    @abstractmethod
    def __init__(self, dataset: Dataset, indices: List[int]) -> None:
        """Class constructor

        Args:
            dataset: Dataset from which to take the subset.
            indices: Indices in the whole set selected for subset.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Gets the length of the subset.

        Returns:
            Length of the subset.
        """

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Gets an item from the subset.

        Args:
            idx: Index of the item in the subset.

        Raises:
            FedbiomedError: if `idx` is out of range

        Returns:
            Item at index `idx` in the subset.
        """


class FrameworkDataManager(ABC):
    """Class for creating data loaders from dataset depending on training plans"""

    _loader_class: Type[DataLoader]
    _loader_arguments: dict
    _dataset_wrapper: Callable

    _dataset: Dataset
    _subset_class: Type[FrameworkSubset]

    _subset_test: Optional[FrameworkSubset] = None
    _subset_train: Optional[FrameworkSubset] = None

    _training_index: List[int] = []
    _testing_index: List[int] = []
    _test_ratio: Optional[float] = None

    @abstractmethod
    def __init__(self, dataset: Dataset, **kwargs: dict):
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader

        Raises:
            FedbiomedError: Bad argument type
        """
        if not isinstance(dataset, Dataset):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `dataset` should be a "
                f"`fedbiomed.common.Dataset` object"
            )
        self._dataset = dataset

        self._loader_arguments = kwargs

    @property
    def dataset(self) -> Dataset:
        """Gets dataset.

        Returns:
            Dataset instance
        """
        return self._dataset

    @abstractmethod
    def _random_split(
        self, dataset: Dataset, lengths: List[int]
    ) -> Tuple[FrameworkSubset, FrameworkSubset]:
        """Randomly split a dataset into 2 non-overlapping subsets of given lengths.

        Args:
            dataset: Dataset to split
            lengths: Lengths of 2 splits to be produced

        Returns:
            List of subsets of the dataset. `None` if the subset is empty.
        """

    def split(
        self,
        test_ratio: float,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Split Dataset into train and validation dataloaders.

        Args:
            test_ratio: Split ratio for validation set ratio. Rest of the samples will be used for training
            test_batch_size: Batch size to use for testing subset
            is_shuffled_testing_dataset: if True, randomly select different samples for the testing
                subset at each execution. If False, reuse previous split when possible.
        Raises:
            FedbiomedError: Arguments bad format
            FedbiomedError: Cannot get number of samples from dataset

        Returns:
            train_loader: DataLoader for training subset. `None` if the `test_ratio` is `1`
            test_loader: DataLoader for validation subset. `None` if the `test_ratio` is `0`
        """
        # No need to check is_shuffled_testing_dataset, any argument can be interpreted as bool

        # Check the type of argument test_batch_size
        if not isinstance(test_batch_size, int) and test_batch_size is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `test_batch_size` should be "
                f"type `int` or `None` not {type(test_batch_size)}"
            )

        # Check the argument `ratio` is of type `float`
        if not isinstance(test_ratio, (float, int)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `test_ratio` should be "
                f"type `float` or `int` not {type(test_ratio)}"
            )

        # Check ratio is valid for splitting
        if test_ratio < 0 or test_ratio > 1:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `test_ratio` should be "
                f"equal or between 0 and 1, not {test_ratio}"
            )

        # Wrap dataset in framework specific class if needed
        framework_dataset = self._dataset_wrapper(self._dataset)

        try:
            samples = len(framework_dataset)
        except AttributeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Can not get number of samples from "
                f"{str(self._dataset)} due to undefined attribute, {str(e)}"
            ) from e
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Can not get number of samples from "
                f"{str(self._dataset)}, {str(e)}"
            ) from e

        if self._test_ratio != test_ratio and self._test_ratio is not None:
            if not is_shuffled_testing_dataset:
                logger.info(
                    "`test_ratio` value has changed: this will change the testing dataset"
                )
            is_shuffled_testing_dataset = True

        _is_loading_failed: bool = False
        # Calculate number of samples for train and validation subsets
        test_samples = math.floor(samples * test_ratio)
        if self._testing_index and not is_shuffled_testing_dataset:
            try:
                self._load_indexes(
                    framework_dataset, self._training_index, self._testing_index
                )
            except IndexError:
                _is_loading_failed = True
        if (
            not self._testing_index or is_shuffled_testing_dataset
        ) or _is_loading_failed:
            train_samples = samples - test_samples

            self._subset_train, self._subset_test = self._random_split(
                framework_dataset,
                [train_samples, test_samples],
            )

            self._testing_index = list(self._subset_test.indices)
            self._training_index = list(self._subset_train.indices)

        if not test_batch_size and self._subset_test is not None:
            test_batch_size = len(self._subset_test)

        self._test_ratio = test_ratio

        loaders = (
            self._subset_loader(self._subset_train, **self._loader_arguments),
            self._subset_loader(self._subset_test, batch_size=test_batch_size),
        )

        return loaders

    def _load_indexes(
        self,
        dataset: Dataset,
        training_index: List[int],
        testing_index: List[int],
    ):
        """Loads training and testing indexes to create subsets.

        Args:
            dataset: Dataset to create subsets from
            training_index: List of indexes for training subset
            testing_index: List of indexes for testing subset
        """
        # Previous checks ensure indexes are valid and within range
        self._subset_train = self._subset_class(dataset, training_index)
        self._subset_test = self._subset_class(dataset, testing_index)

    def save_state(self) -> Dict:
        """Gets state of the data loader.

        Returns:
            A Dict containing data loader state.
        """

        data_manager_state = {}
        data_manager_state["training_index"] = self._training_index
        data_manager_state["testing_index"] = self._testing_index
        data_manager_state["test_ratio"] = self._test_ratio
        return data_manager_state

    def load_state(self, state: Dict):
        """Loads state of the data loader


        It currently keep only testing index, training index and test ratio
        as state.

        Args:
            state: Object containing data loader state.
        """
        self._testing_index = state.get("testing_index", [])
        self._training_index = state.get("training_index", [])
        self._test_ratio = state.get("test_ratio", None)

    @classmethod
    def _subset_loader(
        cls, subset: Optional[FrameworkSubset], **kwargs
    ) -> Optional[DataLoader]:
        """Loads subset (train/validation) partition of as DataLoader.

        Args:
            subset: Subset to create loader
            **kwargs: Loader arguments for DataLoader

        Returns:
            DataLoader for `dataset`. `None` if the subset is empty
        """
        if subset is None or len(subset) <= 0:
            return None

        return cls._create_data_loader(subset, **kwargs)

    @classmethod
    def _create_data_loader(cls, subset: FrameworkSubset, **kwargs: Dict) -> DataLoader:
        """Creates data loader from given subset object

        Args:
            subset: Subset to create loader
            **kwargs: Loader arguments for DataLoader

        Raises:
            FedbiomedError: Raises if DataLoader fails

        Returns:
            Data loader for given dataset
        """

        try:
            # Create a loader from self._dataset to extract inputs and target values
            # by iterating over samples
            loader = cls._loader_class(subset, **kwargs)
        except AttributeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}:  Error while creating DataLoader due to undefined attribute"
                f"{str(e)}"
            ) from e

        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Error while creating DataLoader "
                f"due to incorrect type: {str(e)}"
            ) from e

        return loader
