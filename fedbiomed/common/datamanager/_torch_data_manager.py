# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for Pytorch training plan
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

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
    PytorchDataLoaderItem,
    PytorchDataLoaderSample,
)
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import (
    DatasetDataItem,
    DatasetDataItemModality,
    DataType,
    Transform,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._framework_data_manager import FrameworkDataManager


class _DatasetWrapper(TorchDataset):
    """Wraps a Fed-BioMed dataset in a PyTorch dataset"""

    def __init__(
        self,
        dataset: Dataset,
        is_torch: bool = False,
        transform: Transform = None,
        target_transform: Transform = None,
    ):
        """Class constructor

        Args:
            dataset: Fed-BioMed dataset to wrap
            is_torch: True if data samples returned by `dataset` are already in PyTorch
                format
        """
        self._d = dataset
        self._is_torch = is_torch
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self) -> int:
        """Gets number of samples in dataset

        Returns:
            Length of dataset
        """
        return self._d.__len__()

    @staticmethod
    def _id_function(x: Any) -> Any:
        return x

    def _process_data_item(
        self, data: DatasetDataItem, index: int, is_torch: bool, transform: Transform
    ) -> PytorchDataLoaderItem:
        """Apply conversion and transforms to a data sample data item

        Args:
            data: Data item from data sample
            index: Sequence number of sample in dataset
            is_torch: True if data item is already as PyTorch tensors format
            transform: Transforms to apply to data in PyTorch format
        """
        if data is None:
            return None

        final_data = {}

        for k, v in data.items():
            # Check and convert generic format data
            if not is_torch:
                if not isinstance(v, DatasetDataItemModality):
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Data sample must be a `DatasetDataItemModality` "
                        f"not a {type(v)} (index={index}, modality={k})"
                    )
                if k != v.modality_name:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Data sample modality name mismatch "
                        f"(index={index}), {k} != {v.modality_name}"
                    )
                if v.type == DataType.IMAGE:
                    try:
                        v = torch.from_numpy(v.data)
                    except Exception as e:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB632.value}: Bad data sample format, cannot convert "
                            f"from numpy to torch.Tensor (index={index}, modality={k})"
                        ) from e
                elif v.type == DataType.TABULAR:
                    try:
                        v = torch.tensor(v.data.values)  # type: ignore  # v.data is a pd.DataFrame here
                    except Exception as e:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB632.value}: Bad data sample format, cannot convert "
                            f"from pandas to torch.Tensor (index={index}, modality={k})"
                        ) from e
                else:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Unexpected DataType found."
                    )

            # Find transform function to apply from possible `Transform` cases
            if callable(transform):
                used_transform = transform
            elif isinstance(transform, dict) and k in transform:
                used_transform = transform[k]
            else:
                # In that case, no copy is made - same object
                used_transform = self._id_function

            try:
                final_data[k] = used_transform(v)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Bad framework transform for sample "
                    f"(index={index}), cannot apply to modality={k} in torch.Tensor format"
                ) from e

        return final_data

    def __getitem__(self, index) -> PytorchDataLoaderSample:
        """Get one sample from dataset

        First converts sample from generic sample format to format expected in PyTorch
        training plan. Then applies framework transforms to data and targets.

        Args:
            index: Sequence number of sample in dataset

        Returns:
            A dataset sample
        """
        data, target = self._d.__getitem__(index)

        if data is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Data sample cannot be empty for dataset "
                f"{self._d.__class__.__name__} (index={index})"
            )
        elif not isinstance(data, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad data sample type for dataset "
                f"{self._d.__class__.__name__} (index={index}). Must be `dict` not "
                f"{type(data)}"
            )
        else:
            final_data = self._process_data_item(
                data, index, self._is_torch, self._transform
            )

        if not isinstance(target, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad target sample type for dataset "
                f"{self._d.__class__.__name__} (index={index}). Must be `dict` not "
                f"{type(target)}"
            )
        else:
            final_target = self._process_data_item(
                target, index, self._is_torch, self._target_transform
            )

        return final_data, final_target


class TorchDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for Pytorch training plans"""

    _loader_arguments: dict
    _dataset: Dataset
    _subset_train: Optional[TorchSubset] = None
    _subset_test: Optional[TorchSubset] = None
    _training_index: List[int] = []
    _testing_index: List[int] = []
    _test_ratio: Optional[float] = None
    _to_torch: bool
    _framework_transform: Transform
    _framework_target_transform: Transform

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
        self.rng(self._loader_arguments.get("random_state"))

        self._framework_transform = self._dataset.framework_transform
        # check self._framework_transform is a DatasetDataItem
        if (
            self._framework_transform is not None
            and not callable(self._framework_transform)
            and (
                not isinstance(self._framework_transform, dict)
                or not all([callable(v) for _, v in self._framework_transform.items()])
            )
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad type for argument `framework_transform`. "
                f"Must be a `Transform` not a {type(self._framework_transform)}"
            )

        self._framework_target_transform = self._dataset.framework_target_transform
        # check self._framework_target_transform is a DatasetDataItem
        if (
            self._framework_target_transform is not None
            and not callable(self._framework_target_transform)
            and (
                not isinstance(self._framework_target_transform, dict)
                or not all(
                    [callable(v) for _, v in self._framework_target_transform.items()]
                )
            )
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad type for argument `framework_target_transform`. "
                f"Must be a `Transform` not a {type(self._framework_target_transform)}"
            )

        self._to_torch = False
        if hasattr(self._dataset, "to_torch"):
            logger.debug(
                f"Dataset of type {self._dataset.__class__.__name__} implements "
                "`to_torch()`. Can request data directly in PyTorch format."
            )
            self._to_torch = self._dataset.to_torch()  # type: ignore  # previously tested that the method exists
        else:
            logger.debug(
                f"Dataset of type {self._dataset.__class__.__name__} doesn't implement "
                "`to_torch()`. Data needs intermediate conversion through generic format."
            )

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
        torch_dataset = _DatasetWrapper(
            self._dataset,
            self._to_torch,
            self._framework_transform,
            self._framework_target_transform,
        )
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
                subset at each execution. If False, re-use previous split when possible.
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
        torch_dataset = _DatasetWrapper(
            self._dataset,
            self._to_torch,
            self._framework_transform,
            self._framework_target_transform,
        )

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
                torch_dataset, [train_samples, test_samples], generator=self.rng()
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

    # See issue 1369 regarding potential bug or improvements
    @staticmethod
    def rng(
        rng: Optional[int] = None, device: Optional[str | torch.device] = None
    ) -> Union[None, torch.Generator]:
        """Random number generator

        Returns:
            None if rng is None else a torch generator.
        """

        return None if rng is None else torch.Generator(device).manual_seed(rng)

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
