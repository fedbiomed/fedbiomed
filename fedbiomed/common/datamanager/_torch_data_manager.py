# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Torch data manager
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTorchDataManagerError
from fedbiomed.common.logger import logger

from ._sklearn_data_manager import SkLearnDataManager


class TorchDataManager(object):
    """Wrapper for PyTorch Dataset to manage loading operations for validation and train."""

    def __init__(self, dataset: Dataset, **kwargs: dict):
        """Construct  of class

        Args:
            dataset: Dataset object for torch.utils.data.DataLoader
            **kwargs: Arguments for PyTorch `DataLoader`

        Raises:
            FedbiomedTorchDataManagerError: If the argument `dataset` is not an instance of `torch.utils.data.Dataset`
        """

        # TorchDataManager should get `dataset` argument as an instance of torch.utils.data.Dataset
        if not isinstance(dataset, Dataset):
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: The attribute `dataset` should an instance "
                f"of `torch.utils.data.Dataset`, please use `Dataset` as parent class for"
                f"your custom torch dataset object"
            )

        self._dataset = dataset
        self._loader_arguments = kwargs

        self._rng = self.rng(self._loader_arguments.get("random_state"))
        self._subset_test: Union[Subset, None] = None
        self._subset_train: Union[Subset, None] = None
        self._is_shuffled_testing_dataset: bool = (
            False  # self._loader_arguments.get('shuffle_testing_dataset', False)
        )
        self.training_index: List[int] = []
        self.testing_index: List[int] = []
        self.test_ratio: Optional[float] = None

    @property
    def dataset(self) -> Dataset:
        """Gets dataset.

        Returns:
            PyTorch dataset instance
        """
        return self._dataset

    def subset_test(self) -> Subset:
        """Gets validation subset of the dataset.

        Returns:
            Validation subset
        """

        return self._subset_test

    def subset_train(self) -> Subset:
        """Gets train subset of the dataset.

        Returns:
            Train subset
        """
        return self._subset_train

    def load_all_samples(self) -> DataLoader:
        """Loading all samples as PyTorch DataLoader without splitting.

        Returns:
            Dataloader for entire datasets. `DataLoader` arguments will be retrieved from the `**kwargs` which
                is defined while initializing the class
        """
        return self._create_torch_data_loader(self._dataset, **self._loader_arguments)

    def split(
        self,
        test_ratio: float,
        test_batch_size: Union[int, None],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
        """Splitting PyTorch Dataset into train and validation.

        Args:
             test_ratio: Split ratio for validation set ratio. Rest of the samples will be used for training
        Raises:
            FedbiomedTorchDataManagerError: If the ratio is not in good format

        Returns:
             train_loader: DataLoader for training subset. `None` if the `test_ratio` is `1`
             test_loader: DataLoader for validation subset. `None` if the `test_ratio` is `0`
        """

        # Check the argument `ratio` is of type `float`
        if not isinstance(test_ratio, (float, int)):
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: The argument `ratio` should be "
                f"type `float` or `int` not {type(test_ratio)}"
            )

        # Check ratio is valid for splitting
        if test_ratio < 0 or test_ratio > 1:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: The argument `ratio` should be "
                f"equal or between 0 and 1, not {test_ratio}"
            )

        # If `Dataset` has proper data attribute
        # try to get shape from self.data
        if not hasattr(self._dataset, "__len__"):
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Can not get number of samples from "
                f"{str(self._dataset)} without `__len__`.  Please make sure "
                f"that `__len__` method has been added to custom dataset. "
                f"This method should return total number of samples."
            )

        try:
            samples = len(self._dataset)
        except AttributeError as e:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Can not get number of samples from "
                f"{str(self._dataset)} due to undefined attribute, {str(e)}"
            )
        except TypeError as e:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Can not get number of samples from "
                f"{str(self._dataset)}, {str(e)}"
            )

        if self.test_ratio != test_ratio and self.test_ratio is not None:
            if not is_shuffled_testing_dataset:
                logger.info(
                    "`test_ratio` value has changed: this will change the testing dataset"
                )
            is_shuffled_testing_dataset = True
        _is_loading_failed: bool = False
        # Calculate number of samples for train and validation subsets
        test_samples = math.floor(samples * test_ratio)
        if self.testing_index and not is_shuffled_testing_dataset:
            try:
                self._load_indexes(self.training_index, self.testing_index)
            except IndexError:
                _is_loading_failed = True
        if (
            not self.testing_index or is_shuffled_testing_dataset
        ) or _is_loading_failed:
            train_samples = samples - test_samples

            self._subset_train, self._subset_test = random_split(
                self._dataset, [train_samples, test_samples], generator=self.rng()
            )

            self.testing_index = self._subset_test.indices
            self.training_index = self._subset_train.indices

        if not test_batch_size:

            test_batch_size = len(self._subset_test)

        self.test_ratio = test_ratio

        loaders = (
            self._subset_loader(self._subset_train, **self._loader_arguments),
            self._subset_loader(self._subset_test, batch_size=test_batch_size),
        )

        return loaders

    def to_sklearn(self) -> SkLearnDataManager:
        """Converts PyTorch `Dataset` to sklearn data manager of Fed-BioMed.

        Returns:
            Data manager to use in SkLearn base training plans
        """

        loader = self._create_torch_data_loader(
            self._dataset, batch_size=len(self._dataset)
        )
        # Iterate over samples and get input variable and target variable
        inputs = next(iter(loader))[0].numpy()
        target = next(iter(loader))[1].numpy()
        sklearn_data_manager = SkLearnDataManager(
            inputs=inputs, target=target, **self._loader_arguments
        )
        sklearn_data_manager.testing_index = self.testing_index
        sklearn_data_manager.training_index = self.training_index
        sklearn_data_manager.test_ratio = self.test_ratio
        return sklearn_data_manager

    def _subset_loader(self, subset: Subset, **kwargs) -> Union[DataLoader, None]:
        """Loads subset (train/validation) partition of as pytorch DataLoader.

        Args:
            subset: Subset as an instance of PyTorch's `Subset`

        Returns:
            Data loader for `subset`. `None` if the subset is empty
        """

        # Return None if subset has no data
        if len(subset) <= 0:
            return None

        return self._create_torch_data_loader(subset, **kwargs)

    @staticmethod
    def rng(
        rng: Optional[int] = None,
        device: Optional[str | torch.device] = None
    ) -> Union[None, torch.Generator]:
        """Random number generator

        Returns:
            None if rng is None else a torch generator.
        """

        return None if rng is None else torch.Generator(device).manual_seed(rng)

    def _load_indexes(self, training_index: List[int], testing_index: List[int]):

        self.testing_index = testing_index
        self.training_index = training_index

        # TODO: catch INdexOutOfRange kind of errors
        self._subset_train = Subset(self._dataset, training_index)
        self._subset_test = Subset(self._dataset, testing_index)

    def save_state(self) -> Dict:
        """Gets state of the data loader.

        Returns:
            A Dict containing data loader state.
        """

        data_manager_state = {}
        data_manager_state["training_index"] = self.training_index
        data_manager_state["testing_index"] = self.testing_index
        data_manager_state["test_ratio"] = self.test_ratio
        return data_manager_state

    def load_state(self, state: Dict):
        """Loads state of the data loader


        It currently keep only testing index, training index and test ratio
        as state.

        Args:
            state: Object containing data loader state.
        """
        self.testing_index = state.get("testing_index", [])
        self.training_index = state.get("training_index", [])
        self.test_ratio = state.get("test_ratio", None)

    @staticmethod
    def _create_torch_data_loader(dataset: Dataset, **kwargs: Dict) -> DataLoader:
        """Creates python data loader by given dataset object

        Args:
            dataset: Dataset to create loader
            **kwargs: Loader arguments for PyTorch DataLoader

        Raises:
            FedbiomedTorchDataManagerError: Raises if DataLoader of PyTorch fails

        Returns:
            Data loader for given dataset
        """

        try:
            # Create a loader from self._dataset to extract inputs and target values
            # by iterating over samples
            loader = DataLoader(dataset, **kwargs)
        except AttributeError as e:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}:  Error while creating Torch DataLoader due to undefined attribute"
                f"{str(e)}"
            ) from e

        except TypeError as e:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Error while creating a PyTorch DataLoader "
                f"due to incorrect type: {str(e)}"
            ) from e

        return loader
