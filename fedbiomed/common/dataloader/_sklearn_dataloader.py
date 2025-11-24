# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Class for data loader in PyTorch training plans
"""

from typing import List, Optional, Tuple

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError

from ._dataloader import DataLoader

# Base type for data sample. It is not used as data is returned as batches of samples.
# A sample is tuple `(SkLearnDataLoaderItem, SkLearnDataLoaderItem)` for `(data, target)`
#
# SkLearnDataLoaderItem = Optional[np.ndarray]
# SkLearnDataLoaderSample = Tuple[SkLearnDataLoaderItem, SkLearnDataLoaderItem]

# Type for a batch of samples returned by `PytorchDataLoader` iterator
# A batch is tuple `(SkLearnDataLoaderItemBatch, SkLearnDataLoaderItemBatch)` for `(data, target)`
#
# Caveat ! np.ndarray in a batch have *one more dimension* than a single sample
SkLearnDataLoaderItemBatch = Optional[np.ndarray]
SkLearnDataLoaderSampleBatch = Tuple[
    SkLearnDataLoaderItemBatch, SkLearnDataLoaderItemBatch
]


class SkLearnDataLoader(DataLoader):
    """Data loader class for scikit-learn training plan

    Assumes that fixing seed for reproducibility is handled globally in a calling class
    """

    _dataset: Dataset
    _batch_size: int
    _shuffle: bool
    _drop_last: bool

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Class constructor

        Args:
            dataset: dataset object
            batch_size: batch size for each iteration
            shuffle: True if shuffling before iteration
            drop_last: whether to drop the last batch in case it does not fill the whole batch size

        Raises:
            FedbiomedError: bad argument type or value
        """
        # Note: fixing seed for reproducibility is handled globally in SkLearnDataManager

        # `dataset` type was already checked in SKLearnDataManager, but not kwargs

        if not isinstance(batch_size, int):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad type for `batch_size` argument, "
                f"expected `int` got {type(batch_size)}"
            )
        if batch_size <= 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}. Wrong value for `batch_size` argument, "
                f"expected a non-zero positive integer, got value {batch_size}."
            )
        if not isinstance(shuffle, bool):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad type for `shuffle` argument, "
                f"expected `bool` got {type(shuffle)}"
            )
        if not isinstance(drop_last, bool):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad type for `drop_last` argument, "
                f"expected `bool` got {type(drop_last)}"
            )

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def __len__(self) -> int:
        """Returns the number of batches of the encapsulated dataset"""
        n = len(self._dataset) // self._batch_size
        if not self._drop_last and self.n_remainder_samples() != 0:
            n += 1
        return n

    def __iter__(self) -> "_SkLearnBatchIterator":
        """Returns an iterator over batches of data"""
        return _SkLearnBatchIterator(self)

    @property
    def dataset(self) -> Dataset:
        """Returns the encapsulated dataset

        This needs to be a property to harmonize the API with torch.DataLoader, enabling us to write
        generic code for DataLoaders.
        """
        return self._dataset

    def batch_size(self) -> int:
        """Returns the batch size"""
        return self._batch_size

    def drop_last(self) -> bool:
        """Returns the boolean drop_last attribute"""
        return self._drop_last

    # Specific methods for SkLearnDataLoader

    def shuffle(self) -> bool:
        """Returns the boolean shuffle attribute"""
        return self._shuffle

    def n_remainder_samples(self) -> int:
        """Returns the remainder of the division between dataset length and batch size."""
        return len(self._dataset) % self._batch_size


class _SkLearnBatchIterator:
    """Iterator over batches for SkLearnDataLoader.

    Assumes that fixing seed for reproducibility is handled globally in a calling class

    Attributes:
        _loader: the data loader that created this iterator
        _index: an array  of indices of the samples of the dataset
        _num_yielded: the number of batches yielded in the current epoch
    """

    _loader: SkLearnDataLoader
    _index: np.ndarray
    _num_yielded: int

    def __init__(self, loader: SkLearnDataLoader):
        """Class constructor

        Arguments:
            loader: an instance of the SkLearnDataLoader associated with this iterator
        """
        self._loader = loader
        self._is_initialized = False
        self._is_simple_sample = True
        self._has_target = True
        self._data_keys = None
        self._target_keys = None
        self._reset()

    def _reset(self) -> None:
        """Resets the iterator between epochs.

        Restores num_yielded to 0, reshuffles the indices if shuffle is True, and applies drop_last
        """
        self._num_yielded = 0
        dlen = len(self._loader.dataset)

        self._index = np.arange(dlen)

        # Perform the optional shuffling.
        # Nota: use np.random, to ensure reproducibility with numpy seed setting
        if self._loader.shuffle():
            np.random.shuffle(self._index)

        # Optionally drop the last samples if they make for a smaller batch.
        if self._loader.drop_last() and self._loader.n_remainder_samples() != 0:
            self._index = self._index[: -self._loader.n_remainder_samples()]

    def _initialize(self, data: DatasetDataItem, target: DatasetDataItem) -> None:
        """Initializes the iterator based on the first sample read from the dataset.

        Fixes some expected format for all the samples in the epochs: simple sample or dict of 1 modality,
        is there a target or not, name of the modality if any.

        Args:
            data: a data sample read from the dataset
            target: the corresponding target sample read from the dataset

        Raises:
            FedbiomedError: sample is different from possible formats
        """
        self._is_initialized = True
        if target is None:
            self._has_target = False
        if not isinstance(data, np.ndarray):
            self._is_simple_sample = False
            if isinstance(data, dict):
                self._data_keys = list(data.keys())
                if len(self._data_keys) != 1:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Bad data sample type for dataset "
                        f"(index=0). Expected `np.ndarray` or `Dict[str, np.ndarray]` "
                        f"of 1 modality, got Dict of {len(self._data_keys)} modalities"
                    )
            if isinstance(target, dict):
                self._target_keys = list(target.keys())
                if len(self._target_keys) != 1:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Bad target sample type for dataset "
                        f"(index=0). Expected `np.ndarray` or `Dict[str, np.ndarray]` "
                        f"of 1 modality, got Dict of {len(self._target_keys)} modalities"
                    )

    def _more_info_bad_data_type(self, data: DatasetDataItem, data_keys: List) -> str:
        """Returns a string with more information about a bad data sample type.

        Args:
            data: a data sample read from the dataset
            data_keys: expected data keys if data is a dict

        Returns:
            A string with more information about the bad data sample type.
        """
        data_type = type(data).__name__
        if self._is_simple_sample and isinstance(data, np.ndarray):
            data_type = f"`np.ndarray` with {data.ndim} dimensions"
        elif not self._is_simple_sample and isinstance(data, dict):
            if len(data) != 1:
                data_type = f"`Dict` with {len(data)} modalities"
            elif list(data.keys()) != data_keys:
                data_type = f"`Dict` with non-matching keys {list(set(data.keys()) ^ set(data_keys))}"
            elif not isinstance(list(data.values())[0], np.ndarray):
                data_type = f"`Dict[str, {type(list(data.values())[0]).__name__}]`"
            else:
                data_type = f"`Dict[str, np.ndarray]` with {list(data.values())[0].ndim} dimensions"
        return data_type

    def _check_sample_format(
        self,
        data: DatasetDataItem,
        target: DatasetDataItem,
        sample_index: int,
    ) -> None:
        """ "
        Checks that a sample read from the dataset has the expected typing,
        geometry, and coherence with previous samples.

        Args:
            data: a data sample read from the dataset
            target: the corresponding target sample read from the dataset
            sample_index: index of the sample in the dataset

        Raises:
            FedbiomedError: sample is different from expected format
        """
        if self._is_simple_sample and isinstance(data, np.ndarray) and data.ndim <= 1:
            pass
        elif (
            not self._is_simple_sample
            and isinstance(data, dict)
            and len(data) == 1
            and list(data.keys()) == self._data_keys
            and isinstance(list(data.values())[0], np.ndarray)
            and list(data.values())[0].ndim <= 1
        ):
            pass
        else:
            data_type = self._more_info_bad_data_type(data, self._data_keys)
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Bad data sample type for dataset "
                f"{self._loader.dataset.__class__.__name__} (index={sample_index}). "
                f"Expected `np.ndarray` or `Dict[str, np.ndarray]` of 1 modality and 1 dimension. "
                f"Got {data_type}"
            )

        if self._has_target is not None:
            if (
                self._is_simple_sample
                and isinstance(target, np.ndarray)
                and (target.ndim <= 1)
            ):
                pass
            elif (
                not self._is_simple_sample
                and isinstance(target, dict)
                and len(target) == 1
                and list(target.keys()) == self._target_keys
                and isinstance(list(target.values())[0], np.ndarray)
                and list(target.values())[0].ndim <= 1
            ):
                pass
            else:
                data_type = self._more_info_bad_data_type(target, self._target_keys)
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Bad target sample type for "
                    f"dataset {self._loader.dataset.__class__.__name__} (index={sample_index}). "
                    f"Expected `np.ndarray` or `Dict[str, np.ndarray]` of 1 modality and 1 dimension. "
                    f"got {data_type}"
                )
        elif target is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Inconsistent target sample type "
                f"for dataset {self._loader.dataset.__class__.__name__} "
                f"(index={sample_index}). Expected `None`, got {type(target).__name__}"
            )

    def _add_sample_to_batch(
        self,
        batch_data: SkLearnDataLoaderItemBatch,
        batch_target: SkLearnDataLoaderItemBatch,
        data: DatasetDataItem,
        target: DatasetDataItem,
        is_first_from_batch: bool,
        sample_index: int,
    ) -> Tuple[SkLearnDataLoaderItemBatch, SkLearnDataLoaderItemBatch]:
        """
        Adds a sample to the current batch under construction.

        Args:
            batch_data: current batch of data under construction
            batch_target: current batch of target under construction
            data: a data sample read from the dataset
            target: the corresponding target sample read from the dataset
            is_first_from_batch: True if this is the first sample added to this batch
            sample_index: index of the sample in the dataset
        """

        def _extract_array(sample: DatasetDataItem, keys: Optional[List]) -> np.ndarray:
            """Extract numpy array from sample and ensure at least 1D shape."""
            return np.atleast_1d(sample if self._is_simple_sample else sample[keys[0]])

        # Extract arrays from samples
        data_array = _extract_array(data, self._data_keys)
        target_array = (
            _extract_array(target, self._target_keys) if self._has_target else None
        )

        # Handle first sample in batch
        if is_first_from_batch:
            batch_data = data_array[np.newaxis, ...]
            batch_target = target_array[np.newaxis, ...] if self._has_target else None
            return batch_data, batch_target

        # Add subsequent samples to batch
        try:
            batch_data = np.vstack((batch_data, data_array[np.newaxis, ...]))  # type: ignore
        except ValueError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: cannot batch data samples "
                f"from dataset {self._loader.dataset.__class__.__name__} "
                f"(index={sample_index}). This may be due to inconsistent sample shapes. "
                f"Details: {e}"
            ) from e

        if self._has_target:
            try:
                batch_target = np.vstack((batch_target, target_array[np.newaxis, ...]))  # type: ignore
            except ValueError as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: cannot batch target samples "
                    f"from dataset {self._loader.dataset.__class__.__name__} "
                    f"(index={sample_index}). This may be due to inconsistent sample shapes. "
                    f"Details: {e}"
                ) from e

        return batch_data, batch_target

    def __next__(self) -> SkLearnDataLoaderSampleBatch:
        """Returns the next batch.

        if no target array was provided to the data loader, it will return (data_batch, None), else it will return
        (data_batch, target_batch).

        Automatically resets the iterator after each epoch.

        Returns:
            A batch of samples.

        Raises:
            StopIteration: when an epoch of data has been exhausted.
            FedbiomedError: when a sample cannot be retrieved from the dataset
            FedbiomedError: when a sample has an unexpected type or format
        """
        if self._num_yielded < len(self._loader):
            start = self._num_yielded * self._loader.batch_size()
            stop = (self._num_yielded + 1) * self._loader.batch_size()
            indices = self._index[start:stop]
            self._num_yielded += 1

            # Cannot slice on a Dataset, so we get each sample one by one
            batch_data: SkLearnDataLoaderItemBatch = None
            batch_target: SkLearnDataLoaderItemBatch = None

            for i in indices:
                try:
                    data, target = self._loader.dataset[i]
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: cannot retrieve sample {i} from dataset: {e}"
                    ) from e

                # First sample read by the iterator fixes the expected format for all the epochs
                if not self._is_initialized:
                    self._initialize(data, target)

                # Check typing of sample + coherence, don't trust input from dataset
                self._check_sample_format(data, target, i)

                # Add sample to batch - maybe can be optimized
                batch_data, batch_target = self._add_sample_to_batch(
                    batch_data, batch_target, data, target, i == indices[0], i
                )

            return batch_data, batch_target  # type: ignore

        # Prepare for next epoch
        self._reset()
        raise StopIteration
