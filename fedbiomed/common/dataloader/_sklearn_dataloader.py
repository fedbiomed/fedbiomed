# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Class for data loader in PyTorch training plans
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError

from ._dataloader import DataLoader

# Base type for data sample. It is not used as data is returned as batches of samples.
# A sample is tuple `(SkLearnDataLoaderItem, SkLearnDataLoaderItem)` for `(data, target)`
#
# SkLearnDataLoaderItem = Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
# SkLearnDataLoaderSample = Tuple[SkLearnDataLoaderItem, SkLearnDataLoaderItem]

# Type for a batch of samples returned by `PytorchDataLoader` iterator
# A batch is tuple `(SkLearnDataLoaderItemBatch, SkLearnDataLoaderItemBatch)` for `(data, target)`
#
# Caveat ! np.ndarray in a batch have *one more dimension* than a single sample
SkLearnDataLoaderItemBatch = Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
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
    """Iterator over batches for NPDataLoader.

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
            loader: an instance of the NPDataLoader associated with this iterator
        """
        self._loader = loader
        self._reset()

    def _reset(self) -> None:
        """Reset the iterator between epochs.

        restore num_yielded to 0, reshuffles the indices if shuffle is True, and applies drop_last
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

            has_target = True
            is_simple_sample = True

            # Cannot slice on a Dataset, so we get each sample one by one
            for i in indices:
                try:
                    data, target = self._loader.dataset[i]
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: cannot retrieve sample {i} from dataset: {e}"
                    ) from e

                # First sample in the batch fixes some geometry for the whole batch
                if i == indices[0]:
                    if target is None:
                        has_target = False
                        batch_target = None
                    if not isinstance(data, np.ndarray):
                        is_simple_sample = False

                # Check typing of sample + coherence, don't trust input from dataset
                if is_simple_sample and isinstance(data, np.ndarray):
                    pass
                elif (
                    not is_simple_sample
                    and isinstance(data, dict)
                    and all(isinstance(v, np.ndarray) for v in data.values())
                ):
                    pass
                else:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Bad data sample type for dataset "
                        f"{self._loader.dataset.__class__.__name__} (index={i}). "
                        f"Expected `np.ndarray` or `Dict[str, np.ndarray]` of 1 dimension."
                        f"got {type(data).__name__}"
                    )

                if has_target is not None:
                    if is_simple_sample and isinstance(target, np.ndarray):
                        pass
                    elif (
                        not is_simple_sample
                        and isinstance(target, dict)
                        and all(isinstance(v, np.ndarray) for v in target.values())
                    ):
                        pass
                    else:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB632.value}: Bad target sample type for "
                            f"dataset {self._loader.dataset.__class__.__name__} "
                            f"(index={i}). Expected `np.ndarray` or `Dict[str, np.ndarray]` of 1 dimension."
                            f"got {type(target).__name__}"
                        )
                elif target is not None:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Inconsistent target sample type "
                        f"for dataset {self._loader.dataset.__class__.__name__} "
                        f"(index={i}). Expected `None`, got {type(target).__name__}"
                    )

                # Add sample to batch - maybe can be optimized
                if i == indices[0]:
                    if is_simple_sample:
                        # The ... vs : syntax is needed to handle samples with 0 dimensions (scalars)
                        batch_data = data[np.newaxis, ...]
                        if has_target:
                            batch_target = target[np.newaxis, ...]
                    else:
                        batch_data = {k: data[k][np.newaxis, ...] for k in data.keys()}
                        if has_target:
                            batch_target = {
                                k: target[k][np.newaxis, ...] for k in target.keys()
                            }
                else:
                    if is_simple_sample:
                        try:
                            batch_data = np.vstack((batch_data, data[np.newaxis, ...]))  # type: ignore
                        except ValueError as e:
                            raise FedbiomedError(
                                f"{ErrorNumbers.FB632.value}: cannot batch data samples "
                                f"from dataset {self._loader.dataset.__class__.__name__} "
                                f"(index={i}). This may be due to inconsistent sample shapes. "
                                f"Details: {e}"
                            ) from e
                        if has_target:
                            try:
                                batch_target = np.vstack(
                                    (batch_target, target[np.newaxis, ...])
                                )  # type: ignore
                            except ValueError as e:
                                raise FedbiomedError(
                                    f"{ErrorNumbers.FB632.value}: cannot batch target samples "
                                    f"from dataset {self._loader.dataset.__class__.__name__} "
                                    f"(index={i}). This may be due to inconsistent sample shapes. "
                                    f"Details: {e}"
                                ) from e
                    else:
                        try:
                            batch_data = {
                                k: np.vstack((batch_data[k], data[k][np.newaxis, ...]))
                                for k in data.keys()
                            }  # type: ignore
                        except ValueError as e:
                            raise FedbiomedError(
                                f"{ErrorNumbers.FB632.value}: cannot batch data samples "
                                f"from dataset {self._loader.dataset.__class__.__name__} "
                                f"(index={i}). This may be due to inconsistent sample shapes. "
                                f"Details: {e}"
                            ) from e
                        if has_target:
                            try:
                                batch_target = {
                                    k: np.vstack(
                                        (batch_target[k], target[k][np.newaxis, ...])
                                    )
                                    for k in target.keys()
                                }
                            except ValueError as e:
                                raise FedbiomedError(
                                    f"{ErrorNumbers.FB632.value}: cannot batch target samples "
                                    f"from dataset {self._loader.dataset.__class__.__name__} "
                                    f"(index={i}). This may be due to inconsistent sample shapes. "
                                    f"Details: {e}"
                                ) from e

            return batch_data, batch_target  # type: ignore

        # Prepare for next epoch
        self._reset()
        raise StopIteration
