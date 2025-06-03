# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data Management for numpy datasets in Fed-BioMed.

This module introduces the NPDataLoader to provide a data loader interface for numpy datasets
that is similar to the interface for torch.
"""


from typing import Optional, Tuple

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import (
    FedbiomedTypeError,
    FedbiomedValueError,
)
from fedbiomed.common.logger import logger


class NPDataLoader:
    """DataLoader for a Numpy dataset.

    This data loader encapsulates a dataset composed of numpy arrays and presents an Iterable interface.
    One design principle was to try to make the interface as similar as possible to a torch.DataLoader.

    Attributes:
        _dataset: (np.ndarray) a 2d array of features
        _target: (np.ndarray) an optional array of target values
        _batch_size: (int) the number of elements in one batch
        _shuffle: (bool) if True, shuffle the data at the beginning of every epoch
        _drop_last: (bool) if True, drop the last batch if it does not contain batch_size elements
        _rng: (np.random.Generator) the random number generator for shuffling
    """

    def __init__(
        self,
        dataset: np.ndarray,
        target: np.ndarray,
        batch_size: int = 1,
        shuffle: bool = False,
        random_seed: Optional[int | np.random.Generator] = None,
        drop_last: bool = False,
    ):
        """Construct numpy data loader

        Args:
            dataset: 2D Numpy array
            target: Numpy array of target values
            batch_size: batch size for each iteration
            shuffle: shuffle before iteration
            random_seed: an optional integer to set the numpy random seed for shuffling. If it equals
                None, then no attempt will be made to set the random seed.
            drop_last: whether to drop the last batch in case it does not fill the whole batch size
        """

        if not isinstance(dataset, np.ndarray) or not isinstance(target, np.ndarray):
            msg = (
                f"{ErrorNumbers.FB609.value}. Wrong input type for `dataset` or `target` in NPDataLoader. "
                f"Expected type np.ndarray for both, instead got {type(dataset)} and"
                f"{type(target)} respectively."
            )
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        # If the researcher gave a 1-dimensional dataset, we expand it to 2 dimensions
        if dataset.ndim == 1:
            dataset = dataset[:, np.newaxis]

        # If the researcher gave a 1-dimensional target, we expand it to 2 dimensions
        if target.ndim == 1:
            target = target[:, np.newaxis]

        if dataset.ndim != 2 or target.ndim != 2:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB609.value}. Wrong shape for `dataset` or `target` in "
                f"NPDataLoader. Expected 2-dimensional arrays, instead got {dataset.ndim}- "
                f"dimensional and {target.ndim}-dimensional arrays respectively."
            )

        if len(dataset) != len(target):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB609.value}. Inconsistent length for `dataset` and `target` "
                f"in NPDataLoader. Expected same length, instead got len(dataset)={len(dataset)}, "
                f"len(target)={len(target)}"
            )

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB609.value}. Wrong value for `batch_size` parameter of "
                f"NPDataLoader. Expected a non-zero positive integer, instead got value {batch_size}."
            )

        if random_seed is not None and not isinstance(
            random_seed, (int, np.random.Generator)
        ):
            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB609.value}. Wrong type for `random_seed` parameter of "
                f"NPDataLoader. Expected int or None, instead got {type(random_seed)}."
            )

        self._dataset = dataset
        self._target = target
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = (
            np.random.default_rng(random_seed)
            if isinstance(random_seed, (int, type(None)))
            else random_seed
        )

    def __len__(self) -> int:
        """Returns the length of the encapsulated dataset"""
        n = len(self._dataset) // self._batch_size
        if not self._drop_last and self.n_remainder_samples() != 0:
            n += 1
        return n

    def __iter__(self) -> "_BatchIterator":
        """Returns an iterator over batches of data"""
        return _BatchIterator(self)

    @property
    def dataset(self) -> np.ndarray:
        """Returns the encapsulated dataset

        This needs to be a property to harmonize the API with torch.DataLoader, enabling us to write
        generic code for both DataLoaders.
        """
        return self._dataset

    @property
    def target(self) -> np.ndarray:
        """Returns the array of target values

        This has been made a property to have a homogeneous interface with the dataset property above.
        """
        return self._target

    def batch_size(self) -> int:
        """Returns the batch size"""
        return self._batch_size

    def rng(self) -> np.random.Generator:
        """Returns the random number generator"""
        return self._rng

    def shuffle(self) -> bool:
        """Returns the boolean shuffle attribute"""
        return self._shuffle

    def drop_last(self) -> bool:
        """Returns the boolean drop_last attribute"""
        return self._drop_last

    def n_remainder_samples(self) -> int:
        """Returns the remainder of the division between dataset length and batch size."""
        return len(self._dataset) % self._batch_size


class _BatchIterator:
    """Iterator over batches for NPDataLoader.

    Attributes:
        _loader: (NPDataLoader) the data loader that created this iterator
        _index: (np.array) an array  of indices into the data loader's data
        _num_yielded: (int) the number of batches yielded in the current epoch
    """

    def __init__(self, loader: NPDataLoader):
        """Constructs the _BatchIterator.

        Arguments:
            loader: (NPDataLoader) an instance of the NPDataLoader associated with this iterator
        """
        self._loader = loader
        self._index = None
        self._num_yielded = 0
        self._reset()

    def _reset(self):
        """Reset the iterator between epochs.

        restore num_yielded to 0, reshuffles the indices if shuffle is True, and applies drop_last
        """
        self._num_yielded = 0
        dlen = len(self._loader.dataset)

        self._index = np.arange(dlen)

        # Perform the optional shuffling.
        if self._loader.shuffle():
            self._loader.rng().shuffle(self._index)

        # Optionally drop the last samples if they make for a smaller batch.
        if self._loader.drop_last() and self._loader.n_remainder_samples() != 0:
            self._index = self._index[: -self._loader.n_remainder_samples()]

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns the next batch.

        if no target array was provided to the data loader, it will return (dataset_batch, None), else it will return
        (features_batch, target_batch).

        Automatically resets the iterator after each epoch.

        Raises:
            StopIteration: when an epoch of data has been exhausted.
        """
        if self._num_yielded < len(self._loader):
            start = self._num_yielded * self._loader.batch_size()
            stop = (self._num_yielded + 1) * self._loader.batch_size()
            indices = self._index[start:stop]
            self._num_yielded += 1
            if self._loader.target is None:
                return self._loader.dataset[indices, :], None
            else:
                return self._loader.dataset[indices, :], self._loader.target[indices]

        # Set index to zero for next epochs
        self._reset()
        raise StopIteration
