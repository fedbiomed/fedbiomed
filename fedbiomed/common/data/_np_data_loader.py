from typing import Optional
import numpy as np


class NPDataLoader:
    """DataLoader for Numpy dataset.

    This data loader encapsulates a numpy array, and presents an interface for iterating over the dataset, potentially
    more than once, with or without shuffling. Each iteration consists of a number of samples equal to the batch
    size.
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
            batch_size: (int) Batch size for each iteration
            shuffle: (bool) Shuffle before iteration
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

        self.dataset = dataset
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_seed = random_seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return _BatchIterator(self)

    def num_batches(self):
        n = len(self) // self.batch_size
        if not self.drop_last and len(self) % self.batch_size != 0:
            n += 1
        return n


class _BatchIterator:
    """ Iterator class for batch iteration"""
    def __init__(self, loader: NPDataLoader):
        self._loader = loader
        self._index = None
        self._num_yielded = 0
        self._reset()

        if self._loader.random_seed is not None:
            # Set random seed. This may clash if we set the seed somewhere else as well,
            # but we will improve it when we decide to start using a newer version of scikit-learn
            np.random.seed(self._loader.random_seed)

    def _reset(self):
        self._num_yielded = 0
        dlen = len(self._loader)

        self._index = np.arange(dlen)

        # Perform the optional shuffling.
        if self._loader.shuffle:
            np.random.shuffle(self._index)

        # Optionally drop the last samples if they make for a smaller batch.
        num_remainder_samples = dlen % self._loader.batch_size
        if self._loader.drop_last and num_remainder_samples != 0:
            self._index = self._index[:-num_remainder_samples]

    def __next__(self):
        """Returns the next value from the NPDataLoader"""
        if self._num_yielded < self._loader.num_batches():
            start = self._num_yielded*self._loader.batch_size
            stop = (self._num_yielded+1)*self._loader.batch_size
            indices = self._index[start:stop]
            self._num_yielded += 1
            if self._loader.target is None:
                return self._loader.dataset[indices, :], None
            else:
                return self._loader.dataset[indices, :], self._loader.target[indices, :]

        # Set index to zero for next epochs
        self._reset()
        raise StopIteration
