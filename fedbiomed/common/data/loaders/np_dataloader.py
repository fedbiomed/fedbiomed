from typing import Tuple, Optional
import numpy as np

from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.constants import ErrorNumbers
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

    def __init__(self,
                 dataset: np.ndarray,
                 target: np.ndarray,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 random_seed: Optional[int] = None,
                 drop_last: bool = False):
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
            msg = f"{ErrorNumbers.FB609.value}. Wrong input type for `dataset` or `target` in NPDataLoader. " \
                  f"Expected type np.ndarray for both, instead got {type(dataset)} and" \
                  f"{type(target)} respectively."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        # If the researcher gave a 1-dimensional dataset, we expand it to 2 dimensions
        if dataset.ndim == 1:
            logger.info(f"NPDataLoader expanding 1-dimensional dataset to become 2-dimensional.")
            dataset = dataset[:, np.newaxis]

        # If the researcher gave a 1-dimensional target, we expand it to 2 dimensions
        if target.ndim == 1:
            logger.info(f"NPDataLoader expanding 1-dimensional target to become 2-dimensional.")
            target = target[:, np.newaxis]

        if dataset.ndim != 2 or target.ndim != 2:
            msg = f"{ErrorNumbers.FB609.value}. Wrong shape for `dataset` or `target` in NPDataLoader. " \
                  f"Expected 2-dimensional arrays, instead got {dataset.ndim}-dimensional " \
                  f"and {target.ndim}-dimensional arrays respectively."
            logger.error(msg)
            raise FedbiomedValueError(msg)

        if len(dataset) != len(target):
            msg = f"{ErrorNumbers.FB609.value}. Inconsistent length for `dataset` and `target` in NPDataLoader. " \
                  f"Expected same length, instead got len(dataset)={len(dataset)}, len(target)={len(target)}"
            logger.error(msg)
            raise FedbiomedValueError(msg)

        if not isinstance(batch_size, int):
            msg = f"{ErrorNumbers.FB609.value}. Wrong type for `batch_size` parameter of NPDataLoader. Expected a " \
                  f"non-zero positive integer, instead got type {type(batch_size)}."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        if batch_size <= 0:
            msg = f"{ErrorNumbers.FB609.value}. Wrong value for `batch_size` parameter of NPDataLoader. Expected a " \
                  f"non-zero positive integer, instead got value {batch_size}."
            logger.error(msg)
            raise FedbiomedValueError(msg)

        if not isinstance(shuffle, bool):
            msg = f"{ErrorNumbers.FB609.value}. Wrong type for `shuffle` parameter of NPDataLoader. Expected `bool`, " \
                  f"instead got {type(shuffle)}."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        if not isinstance(drop_last, bool):
            msg = f"{ErrorNumbers.FB609.value}. Wrong type for `drop_last` parameter of NPDataLoader. " \
                  f"Expected `bool`, instead got {type(drop_last)}."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        if random_seed is not None and not isinstance(random_seed, int):
            msg = f"{ErrorNumbers.FB609.value}. Wrong type for `random_seed` parameter of NPDataLoader. " \
                  f"Expected int or None, instead got {type(random_seed)}."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        self._dataset = dataset
        self._target = target
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        """Returns the length of the encapsulated dataset"""
        n = len(self._dataset) // self._batch_size
        if not self._drop_last and self.n_remainder_samples() != 0:
            n += 1
        return n

    def __iter__(self) -> '_BatchIterator':
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
    """ Iterator over batches for NPDataLoader.

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
        self._last_idx_yielded = 0
        self._reset()

    def _reset(self):
        """Reset the iterator between epochs.

        restore num_yielded to 0, reshuffles the indices if shuffle is True, and applies drop_last
        """
        self._last_idx_yielded = 0
        dlen = len(self._loader.dataset)

        self._index = np.arange(dlen)

        # Perform the optional shuffling.
        if self._loader.shuffle():
            self._loader.rng().shuffle(self._index)

        # Optionally drop the last samples if they make for a smaller batch.
        if self._loader.drop_last() and self._loader.n_remainder_samples() != 0:
            self._index = self._index[:-self._loader.n_remainder_samples()]

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns the next batch.

        if no target array was provided to the data loader, it will return (dataset_batch, None), else it will return
        (features_batch, target_batch).

        Automatically resets the iterator after each epoch.
        """
        start = self._last_idx_yielded
        stop = self._last_idx_yielded + self._loader.batch_size()
        if stop > len(self._index):
            indices = self._index[start:]
            self._reset()
            stop = stop - len(self._index)
            indices = np.concatenate((indices, self._index[:stop]))
        else:
            indices = self._index[start:stop]

        self._last_idx_yielded = stop
        if stop >= len(self._index):
            self._reset()

        if self._loader.target is None:
            return self._loader.dataset[indices, :], None
        else:
            return self._loader.dataset[indices, :], self._loader.target[indices, :]


