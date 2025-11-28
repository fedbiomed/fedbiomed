# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for Pytorch training plan
"""

from typing import List, Optional, Tuple, Type

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

from ._framework_data_manager import FrameworkDataManager, FrameworkSubset


class _TorchSubset(TorchSubset, FrameworkSubset):
    pass


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

    def _more_info(self, data) -> str:
        """Generates more info about bad data sample type
        Args:
            data: data sample
        Returns:
            More info string
        """
        if not isinstance(data, dict):
            more_info = f" type {str(type(data))}"
        else:
            more_info = "dict with modalities " + ", ".join(
                [
                    f"`{k}` type `{str(type(v))}`"
                    for k, v in data.items()
                    if not isinstance(v, torch.Tensor)
                ]
            )
        return more_info

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
                f"{self._d.__class__.__name__} (index={index}). Must be `dict of torch.Tensor` or `torch.Tensor` "
                f"not: {self._more_info(data)}"
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
                f"{self._d.__class__.__name__} (index={index}). Must be `dict of torch.Tensor` or `torch.Tensor` "
                f"not: {self._more_info(target)}"
            )

        return data, target


class TorchDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for Pytorch training plans"""

    _loader_class = PytorchDataLoader
    _subset_class = _TorchSubset
    _dataset_wrapper = _DatasetWrapper

    # Better type hinting
    _subset_train: Optional[_TorchSubset] = None
    _subset_test: Optional[_TorchSubset] = None
    _loader_class: Type[PytorchDataLoader]

    def __init__(self, dataset: Dataset, **kwargs: dict):
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader

        Raises:
            FedbiomedError: Bad argument type
        """
        super().__init__(dataset, **kwargs)

        # Note: managing seed to control reproducibility is now done in training plan
        # `post_init` method.
        # Randomization for torch data manager & loader uses only `torch.manual_seed()`

        self._dataset.to_format = DataReturnFormat.TORCH

    # Nota: used only for unit tests
    def subset_test(self) -> Optional[_TorchSubset]:
        """Gets validation subset of the dataset.

        Returns:
            Validation subset
        """

        return self._subset_test

    # Nota: used only for unit tests
    def subset_train(self) -> Optional[_TorchSubset]:
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
        return self._create_data_loader(torch_dataset, **self._loader_arguments)  # type: ignore

    def _random_split(
        self, dataset: Dataset, lengths: List[int]
    ) -> Tuple[_TorchSubset, _TorchSubset]:
        """Randomly split a dataset into 2 non-overlapping subsets of given lengths.

        Args:
            dataset: Dataset to split
            lengths: Lengths of 2 splits to be produced

        Returns:
            List of subsets of the dataset. `None` if the subset is empty.
        """
        return random_split(dataset, lengths)  # type: ignore[return-value]
