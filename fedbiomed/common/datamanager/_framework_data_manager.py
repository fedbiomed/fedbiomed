# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Generic data manager
"""

import math
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloader import DataLoader
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._split_spec import SplitSpec

# WrappedDatasetT is the framework-specific wrapped dataset type produced by _dataset_wrapper
# (e.g. TorchDataset for Torch, fedbiomed Dataset for SkLearn).
WrappedDatasetT = TypeVar("WrappedDatasetT")


class FrameworkSubset(ABC, Generic[WrappedDatasetT]):
    """Subset of a dataset at specified indices."""

    # Sequence (not List) to stay compatible with TorchSubset.indices which is Sequence[int]
    indices: Sequence[int]

    @abstractmethod
    def __init__(self, dataset: WrappedDatasetT, indices: List[int]) -> None:
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


class FrameworkDataManager(ABC, Generic[WrappedDatasetT]):
    """Class for creating data loaders from dataset depending on training plans"""

    _loader_class: Type[DataLoader]
    _loader_arguments: dict
    _dataset_wrapper: Callable

    _dataset: Dataset
    _subset_class: Type[FrameworkSubset[WrappedDatasetT]]

    _subset_validation: Optional[FrameworkSubset[WrappedDatasetT]] = None
    _subset_train: Optional[FrameworkSubset[WrappedDatasetT]] = None

    _training_indices: List[int] = []
    _validation_indices: List[int] = []
    _split_args: Dict[str, object] = {}
    _split_method: Optional[str] = None
    _split_spec: Optional[SplitSpec] = None

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

    def init_split_spec(self, split_spec: Optional[SplitSpec]) -> None:
        """Initializes the DataManager's split specification

        Args:
            split_spec: the split specification to initialize

        Raises:
            FedbiomedError: if `_split_spec` is already initialized
        """
        if self._split_spec is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Split specification is already initialized."
            )
        self._split_spec = split_spec

    @property
    def split_spec(self) -> Optional[SplitSpec]:
        """Gets split_spec.

        Returns:
            SplitSpec instance
        """
        return self._split_spec

    @property
    def has_split_spec(self) -> bool:
        """Check if _split_spec attribute exists

        Returns:
            A boolean set to True if _split_spec attribute is not None
        """
        return self._split_spec is not None

    @abstractmethod
    def _random_split(
        self, dataset: WrappedDatasetT, lengths: List[int]
    ) -> Tuple[FrameworkSubset[WrappedDatasetT], FrameworkSubset[WrappedDatasetT]]:
        """Randomly split a dataset into 2 non-overlapping subsets of given lengths.

        Args:
            dataset: Dataset to split
            lengths: Lengths of 2 splits to be produced

        Returns:
            List of subsets of the dataset. `None` if the subset is empty.
        """

    def split(
        self,
        split_arguments: dict,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Split Dataset into train and validation dataloaders. Either use specified split_spec
        if provided by the user in training plan, or default splitting method.

        Args:
            split_arguments: Dictionary containing splitting arguments.
            test_batch_size: Batch size to use for testing subset
            is_shuffled_testing_dataset: if True, randomly select different samples for the testing
                subset at each execution, in case of default splitting method.
                If False, reuse previous split when possible.
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

        # WrappedDatasetTrap dataset in framework specific class if needed
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

        # No safe way to recover the splits from previous rounds:
        # it is not because split args are the same from one round to another
        # that the splits are the same, due to some possible randomness in
        # init_slit_spec function.
        # If split specifications are provided, use these indices
        if self.has_split_spec:
            # If split specifications are provided, use them to create subsets
            self._training_indices = self.split_spec.train_indices
            self._validation_indices = self.split_spec.validation_indices
            self._split_args = split_arguments
            self._split_method = "custom"
            self._subset_train = self._subset_class(
                framework_dataset, self._training_indices
            )
            self._subset_validation = self._subset_class(
                framework_dataset, self._validation_indices
            )

        # If split specifications are not provided, try default splitting behaviour
        else:
            # Get test ratio from saved state if available
            test_ratio_from_state = None
            # Avoid retrieving a `test_ratio` argument from a custom splitting method used in previous round.
            if self._split_method == "default":
                test_ratio_from_state = self._split_args.get("test_ratio", None)

            # Get test_ratio from split_arguments
            test_ratio = split_arguments.get("test_ratio")

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

            # Check if dataset splits must change from previous round due to change in
            # test_ratio argument
            if (
                test_ratio_from_state != test_ratio
                and test_ratio_from_state is not None
            ):
                if not is_shuffled_testing_dataset:
                    logger.info(
                        "The value of `test_ratio` argument for default dataset splitting "
                        "has changed from previous round or has not been provided: the testing "
                        "dataset will be reshuffled"
                    )
                is_shuffled_testing_dataset = True

            _is_loading_failed: bool = False
            # Calculate number of samples for train and validation subsets
            test_samples = math.floor(samples * test_ratio)
            train_samples = samples - test_samples

            # Do not use validation_indices from previous round generated from custom splitting method
            if (
                self._split_method == "default"
                and self._validation_indices
                and not is_shuffled_testing_dataset
            ):
                try:
                    self._load_indexes(
                        framework_dataset,
                        self._training_indices,
                        self._validation_indices,
                    )
                    _is_loading_failed = False
                except IndexError:
                    _is_loading_failed = True

            # Need new split if the previous round was not using a default splitting method
            # or if the previous validation indices are empty
            # or it is explicitly requested to shuffle the testing dataset
            # or if loading the previous indices failed
            need_new_split = (
                self._split_method != "default"
                or not self._validation_indices
                or is_shuffled_testing_dataset
                or _is_loading_failed
            )

            if need_new_split:
                if self._loader_arguments.get("shuffle", True):
                    # Random split (shuffled)
                    self._subset_train, self._subset_validation = self._random_split(
                        framework_dataset,
                        [train_samples, test_samples],
                    )
                else:
                    # Deterministic split (no shuffle) — preserve original order
                    all_indices = list(range(samples))
                    train_indices = all_indices[:train_samples]
                    test_indices = all_indices[
                        train_samples : train_samples + test_samples
                    ]

                    self._subset_train = self._subset_class(
                        framework_dataset, train_indices
                    )
                    self._subset_validation = (
                        self._subset_class(framework_dataset, test_indices)
                        if test_samples > 0
                        else None
                    )

                self._training_indices = list(self._subset_train.indices)
                self._validation_indices = (
                    list(self._subset_validation.indices)
                    if self._subset_validation is not None
                    else []
                )

            # Update split arguments
            self._split_args = split_arguments
            self._split_method = "default"

        if not test_batch_size and self._subset_validation is not None:
            test_batch_size = len(self._subset_validation)

        loaders = (
            self._subset_loader(self._subset_train, **self._loader_arguments),
            self._subset_loader(self._subset_validation, batch_size=test_batch_size),
        )

        return loaders

    def _load_indexes(
        self,
        dataset: WrappedDatasetT,
        training_indices: List[int],
        validation_indices: List[int],
    ):
        """Loads training and validation indices to create subsets.

        Args:
            dataset: Dataset to create subsets from
            training_indices: List of indexes for training subset
            validation_indices: List of indexes for validation subset
        """
        # Previous checks ensure indexes are valid and within range
        self._subset_train = self._subset_class(dataset, training_indices)
        self._subset_validation = self._subset_class(dataset, validation_indices)

    def save_state(self) -> Dict:
        """Gets state of the data loader (training and validation indices)

        Returns:
            A Dict containing data loader state.
        """

        data_manager_state: Dict[str, object] = {}
        data_manager_state["split_indices"] = {}
        data_manager_state["split_indices"]["training_indices"] = self._training_indices
        data_manager_state["split_indices"]["validation_indices"] = (
            self._validation_indices
        )
        data_manager_state["split_args"] = self._split_args
        data_manager_state["split_method"] = self._split_method
        return data_manager_state

    def load_state(self, state: Dict):
        """Loads state of the data loader

        It currently keeps validation indices, training indices, split arguments,
        and split method as state.

        Args:
            state: Object containing data loader state.
        """
        split_indices = state.get("split_indices", dict())
        self._validation_indices = split_indices.get("validation_indices", [])
        self._training_indices = split_indices.get("training_indices", [])
        self._split_args = state.get("split_args", None)
        self._split_method = state.get("split_method", None)

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
