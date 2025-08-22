# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for Pytorch training plan
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import (
    Dataset as TorchDataset,
)
from torch.utils.data import (
    Subset as TorchSubset,
)
from torch.utils.data import (
    random_split,
)

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloader import (
    PytorchDataLoader,
    PytorchDataLoaderSample,
)
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._framework_data_manager import FrameworkDataManager


class _DatasetWrapper(TorchDataset):
    """Wraps a Fed-BioMed dataset in a PyTorch dataset"""

    def __init__(
        self,
        dataset: Dataset,
    ):
        """Class constructor

        Args:
            dataset: Fed-BioMed dataset to wrap
        """
        self._d = dataset

    def __len__(self) -> int:
        """Gets number of samples in dataset

        Returns:
            Length of dataset
        """
        return self._d.__len__()

    def __getitem__(self, index) -> PytorchDataLoaderSample:
        """Gets one sample from dataset

        Also checks sample format.

        Args:
            index: Sequence number of sample in dataset

        Returns:
            A dataset sample
        """
        data, target = self._d.__getitem__(index)

        if isinstance(data, torch.Tensor):
            pass
        elif isinstance(data, dict) and all(
            isinstance(v, torch.Tensor) for v in data.values()
        ):
            pass
        elif data is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Data sample cannot be empty for dataset "
                f"{self._d.__class__.__name__} (index={index})"
            )
        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad data sample type for dataset "
                f"{self._d.__class__.__name__} (index={index}). Must be `dict` or `torch.Tensor` "
                f"not {type(data)}"
            )

        if isinstance(target, torch.Tensor):
            pass
        elif isinstance(target, dict) and all(
            isinstance(v, torch.Tensor) for v in target.values()
        ):
            pass
        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad target sample type for dataset "
                f"{self._d.__class__.__name__} (index={index}). Must be `dict` or `torch.Tensor` "
                f"not {type(target)}"
            )

        return data, target


class TorchDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for Pytorch training plans"""

    _loader_arguments: dict
    _dataset: Dataset
    _subset_train: Optional[TorchSubset] = None
    _subset_test: Optional[TorchSubset] = None
    _training_index: List[int] = []
    _testing_index: List[int] = []
    _test_ratio: Optional[float] = None

    def __init__(self, dataset: Dataset, **kwargs: dict):  # noqa : B027 # not yet implemented
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader

        Raises:
            FedbiomedError: Bad argument type
        """

        # TorchDataManager should get `dataset` argument as an instance of torch.utils.data.Dataset
        if not isinstance(dataset, Dataset):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `dataset` should an instance "
                f"of `torch.utils.data.Dataset`, please use `Dataset` as parent class for"
                f"your custom torch dataset object"
            )
        self._dataset = dataset

        self._loader_arguments = kwargs

        # Partially address issue 1369 (item 3) to ensure reproducibility:
        # fix seed globally for torch, so it applies to split and shuffle
        # + for all devices
        #
        # Can remove this item as it is not used in PyTorch DataLoader
        seed = self._loader_arguments.pop("random_state", None)
        if isinstance(seed, int):
            torch.manual_seed(seed)

        self._dataset.to_format = DataReturnFormat.TORCH

    @property
    def dataset(self) -> Dataset:
        """Gets dataset.

        Returns:
            Dataset instance
        """
        return self._dataset

    # Nota: used only for unit tests
    def subset_test(self) -> Optional[TorchSubset]:
        """Gets validation subset of the dataset.

        Returns:
            Validation subset
        """

        return self._subset_test

    # Nota: used only for unit tests
    def subset_train(self) -> Optional[TorchSubset]:
        """Gets train subset of the dataset.

        Returns:
            Train subset
        """
        return self._subset_train

    # Nota: used only for unit tests
    def load_all_samples(self) -> PytorchDataLoader:
        """Loading all samples as PyTorch DataLoader without splitting.

        Returns:
            Dataloader for entire datasets. `DataLoader` arguments will be retrieved from the `**kwargs` which
                is defined while initializing the class
        """
        torch_dataset = _DatasetWrapper(self._dataset)
        return self._create_torch_data_loader(torch_dataset, **self._loader_arguments)

    def split(
        self,
        test_ratio: float,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[Optional[PytorchDataLoader], Optional[PytorchDataLoader]]:
        """Splitting PyTorch Dataset into train and validation.

        Args:
            test_ratio: Split ratio for validation set ratio. Rest of the samples will be used for training
            test_batch_size: Batch size to use for testing subset
            is_shuffled_testing_dataset: if True, randomly select different samples for the testing
                subset at each execution. If False, reuse previous split when possible.
        Raises:
            FedbiomedError: Arguments bad format
            FedbiomedError: Cannot get number of samples from dataset

        Returns:
            train_loader: PytorchDataLoader for training subset. `None` if the `test_ratio` is `1`
            test_loader: PytorchDataLoader for validation subset. `None` if the `test_ratio` is `0`
        """
        # No need to check is_shuffled_testing_dataset, amy argument can be interpreted as bool

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

        # Nota: cannot build PyTorch dataset sooner (eg in constructor) because
        # some customization methods may be called in the meantime
        # (cf fedbiomed.node.Round)
        torch_dataset = _DatasetWrapper(self._dataset)

        try:
            samples = len(torch_dataset)
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
                    torch_dataset, self._training_index, self._testing_index
                )
            except IndexError:
                _is_loading_failed = True
        if (
            not self._testing_index or is_shuffled_testing_dataset
        ) or _is_loading_failed:
            train_samples = samples - test_samples

            self._subset_train, self._subset_test = random_split(
                torch_dataset,
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

    def _subset_loader(
        self, subset: Optional[TorchSubset], **kwargs
    ) -> Optional[PytorchDataLoader]:
        """Loads subset (train/validation) partition of as pytorch DataLoader.

        Args:
            subset: Subset as an instance of PyTorch's `Subset`

        Returns:
            PytorchDataLoader for `subset`. `None` if the subset is empty
        """

        # Return None if subset has no data
        if subset is None or len(subset) <= 0:
            return None

        return self._create_torch_data_loader(subset, **kwargs)

    def _load_indexes(
        self,
        torch_dataset: TorchDataset,
        training_index: List[int],
        testing_index: List[int],
    ):
        # Improvement: catch INdexOutOfRange kind of errors
        self._subset_train = TorchSubset(torch_dataset, training_index)
        self._subset_test = TorchSubset(torch_dataset, testing_index)

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

    @staticmethod
    def _create_torch_data_loader(
        dataset: TorchDataset, **kwargs: Dict
    ) -> PytorchDataLoader:
        """Creates torch data loader from given torch dataset object

        Args:
            dataset: Dataset to create loader
            **kwargs: Loader arguments for PyTorch DataLoader

        Raises:
            FedbiomedError: Raises if DataLoader of PyTorch fails

        Returns:
            Data loader for given dataset
        """

        try:
            # Create a loader from self._dataset to extract inputs and target values
            # by iterating over samples
            loader = PytorchDataLoader(dataset, **kwargs)  # type: ignore  # catch errors if kwargs are incorrect
        except AttributeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}:  Error while creating Torch DataLoader due to undefined attribute"
                f"{str(e)}"
            ) from e

        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Error while creating a PyTorch DataLoader "
                f"due to incorrect type: {str(e)}"
            ) from e

        return loader
