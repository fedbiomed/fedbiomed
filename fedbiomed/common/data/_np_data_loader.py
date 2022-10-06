from typing import Optional, Tuple
import numpy as np


class NPDataLoader:
    """DataLoader for Numpy dataset.

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
                 target: Optional[np.ndarray] = None,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 random_seed: Optional[int] = None,
                 drop_last: bool = False):
        """Construct numpy data loader

        Args:
            dataset: (np.ndarray) 2D Numpy array
            target: (Optional[np.ndarray]) 2D Numpy array of target values
            batch_size: (int) batch size for each iteration
            shuffle: (bool) shuffle before iteration
            random_seed: (int or None) an optional integer to set the numpy random seed for shuffling
            drop_last: (bool) whether to drop the last batch in case it does not fill the whole batch size
        """

        if not isinstance(dataset, np.ndarray):
            raise ValueError()

        if target is not None:
            if not isinstance(target, np.ndarray):
                raise ValueError()
            if len(dataset) != len(target):
                raise ValueError()

            # Check target dimensions, we try to be very nice to the researcher
            # First, if they provided an array with too many dimensions, we try to squeeze it
            if len(target.shape) > 2:
                target = target.squeeze()
            # Second, if target was squeezed to 1 dimension or if the researcher gave a 1d target, we expand it
            if len(target.shape) == 1:
                target = target[:, np.newaxis]
            # Finally, if none of the above helped, we raise a ValueError
            if len(target.shape) > 2:
                raise ValueError()

        if len(dataset.shape) > 2:
            raise ValueError

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError()

        if not isinstance(shuffle, bool):
            raise ValueError()

        if not isinstance(drop_last, bool):
            raise ValueError()

        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError()

        self._dataset = dataset
        self._target = target
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        """Returns the length of the encapsulated dataset"""
        return len(self._dataset)

    def __iter__(self) -> '_BatchIterator':
        """Returns an iterator over batches of data"""
        return _BatchIterator(self)

    def get_num_batches(self) -> int:
        """Returns the number of batches in one epoch"""
        n = len(self) // self._batch_size
        if not self._drop_last and len(self) % self._batch_size != 0:
            n += 1
        return n

    def get_dataset(self) -> np.ndarray:
        """Returns the encapsulated dataset"""
        return self._dataset

    def get_target(self) -> np.ndarray:
        """Returns the array of target values"""
        return self._target

    def get_batch_size(self) -> int:
        """Returns the batch size"""
        return self._batch_size

    def get_rng(self) -> np.random.Generator:
        """Returns the random number generator"""
        return self._rng

    def get_shuffle(self) -> bool:
        """Returns the boolean shuffle attribute"""
        return self._shuffle

    def get_drop_last(self) -> bool:
        """Returns the boolean drop_last attribute"""
        return self._drop_last


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
        self._num_yielded = 0
        self._reset()

    def _reset(self):
        """Reset the iterator between epochs.

        restore num_yielded to 0, reshuffles the indices if shuffle is True, and applies drop_last
        """
        self._num_yielded = 0
        dlen = len(self._loader)

        self._index = np.arange(dlen)

        # Perform the optional shuffling.
        if self._loader.get_shuffle():
            self._loader.get_rng().shuffle(self._index)

        # Optionally drop the last samples if they make for a smaller batch.
        num_remainder_samples = dlen % self._loader.get_batch_size()
        if self._loader.get_drop_last() and num_remainder_samples != 0:
            self._index = self._index[:-num_remainder_samples]

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns the next batch.

        if no target array was provided to the data loader, it will return (dataset_batch, None), else it will return
        (features_batch, target_batch).

        Automatically resets the iterator after each epoch.

        Raises:
            StopIteration when an epoch of data has been exhausted.
        """
        if self._num_yielded < self._loader.get_num_batches():
            start = self._num_yielded*self._loader.get_batch_size()
            stop = (self._num_yielded+1)*self._loader.get_batch_size()
            indices = self._index[start:stop]
            self._num_yielded += 1
            if self._loader.get_target() is None:
                return self._loader.get_dataset()[indices, :], None
            else:
                return self._loader.get_dataset()[indices, :], self._loader.get_target()[indices, :]

        # Set index to zero for next epochs
        self._reset()
        raise StopIteration
