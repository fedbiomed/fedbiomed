import numpy as np


class NPDataLoader:

    def __init__(self,
                 dataset: np.ndarray,
                 batch_size: int,
                 shuffle: bool = False,
                 drop_last: bool = False):

        """Construct numpy data loader

        Args:
            dataset: 2D Numpy array
            batch_size: Batch size for each iteration
            shuffle: Shuffle before iteration
        """

        if not isinstance(dataset, np.ndarray):
            raise ValueError()

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError()

        if not isinstance(shuffle, bool):
            raise ValueError()

        if not isinstance(drop_last, bool):
            raise ValueError()

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.size = len(dataset)
        self.num_batches = int(self.size / self.batch_size)
        self.last = self.size % self.batch_size

        if not drop_last:
            self.num_batches += 1

    def get_batch(self, index):
        return self.dataset[index: index+self.batch_size]

    def __iter__(self):
        return _BatchIterator(self)


class _BatchIterator:
    """ Iterator class for batch iteration"""
    def __init__(self, loader):
        self._loader = loader
        self._index = 0

    def __next__(self):
        """Returns the next value from the NPDataLoader"""

        if self._index <= self._loader.num_batches:
            batch = self._loader.get_batch(self._index)
            result = self._index, batch
            self._index += 1

            return result

        # Set index to zero for next epochs
        self._index = 0
        raise StopIteration
