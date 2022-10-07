"""
Data Management for scikit-learn in Fed-BioMed.

This module introduces the NPDataLoader and SkLearnDataManager classes, to provide a data management interface for
Fed-BioMed users relying on the scikit-learn framework that is similar to the interface for torch.
"""


from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_method_spec


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
                 target: Optional[np.ndarray] = None,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 random_seed: Optional[int] = None,
                 drop_last: bool = False):
        """Construct numpy data loader

        Args:
            dataset: (np.ndarray) 2D Numpy array
            target: (Optional[np.ndarray]) Numpy array of target values
            batch_size: (int) batch size for each iteration
            shuffle: (bool) shuffle before iteration
            random_seed: (int or None) an optional integer to set the numpy random seed for shuffling. If it equals
                None, then no attempt will be made to set the random seed.
            drop_last: (bool) whether to drop the last batch in case it does not fill the whole batch size
        """

        if not isinstance(dataset, np.ndarray):
            msg = f"{ErrorNumbers.FB609.value}. Wrong input type for `dataset` in NPDataLoader. Expected type " \
                  f"np.ndarray, instead got {type(dataset)}"
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        if dataset.ndim != 2:
            msg = f"{ErrorNumbers.FB609.value}. Wrong shape for `dataset` in NPDataLoader. Expected 2-dimensional " \
                  f"array, instead got a {dataset.ndim}-dimensional array."
            logger.error(msg)
            raise FedbiomedValueError(msg)

        if target is not None:
            if not isinstance(target, np.ndarray):
                msg = f"{ErrorNumbers.FB609.value}. Wrong type for `target` in NPDataLoader. Expected type " \
                      f"np.ndarray, instead got {type(target)}"
                logger.error(msg)
                raise FedbiomedTypeError(msg)
            if len(dataset) != len(target):
                msg = f"{ErrorNumbers.FB609.value}. Inconsistent length for `dataset` and `target` in NPDataLoader. " \
                      f"Expected same length, instead got len(dataset)={len(dataset)}, len(target)={len(target)}"
                logger.error(msg)
                raise FedbiomedValueError(msg)

            # If the researcher gave a 1-dimensional target, we expand it to 2 dimensions
            if target.ndim == 1:
                logger.warning(f"NPDataLoader :: Expanding 1-dimensional target to become 2-dimensional.")
                target = target[:, np.newaxis]

        if not isinstance(batch_size, int) or batch_size <= 0:
            msg = f"{ErrorNumbers.FB609.value}. Wrong type for `batch_size` parameter of NPDataLoader. Expected a " \
                  f"non-zero positive integer, instead got type {type(batch_size)} with value {batch_size}."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

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
        return len(self._dataset)

    def __iter__(self) -> '_BatchIterator':
        """Returns an iterator over batches of data"""
        return _BatchIterator(self)

    def get_num_batches(self) -> int:
        """Returns the number of batches in one epoch"""
        n = len(self) // self._batch_size
        if not self._drop_last and self.get_n_remainder_samples() != 0:
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

    def get_n_remainder_samples(self) -> int:
        """Returns the remainder of the division between dataset length and batch size."""
        return len(self) % self._batch_size


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
        if self._loader.get_drop_last() and self._loader.get_n_remainder_samples() != 0:
            self._index = self._index[:-self._loader.get_n_remainder_samples()]

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


class SkLearnDataManager(object):
    """Wrapper for `pd.DataFrame`, `pd.Series` and `np.ndarray` datasets.

    Manages datasets for scikit-learn based model training. Responsible for managing inputs, and target
    variables that have been provided in `training_data` of scikit-learn based training plans.
    """
    def __init__(self,
                 inputs: Union[np.ndarray, pd.DataFrame, pd.Series],
                 target: Union[np.ndarray, pd.DataFrame, pd.Series],
                 **kwargs: dict):

        """ Construct a SkLearnDataManager from an array of inputs and an array of targets.

        The loader arguments will be passed to the [fedbiomed.common.data.NPDataLoader] classes instantiated
        when split is called. They may include batch_size, shuffle, drop_last, and others. Please see the
        [fedbiomed.common.data.NPDataLoader] class for more details.

        Args:
            inputs: Independent variables (inputs, features) for model training
            target: Dependent variable/s (target) for model training and validation
            **kwargs: Loader arguments
        """

        if not isinstance(inputs, (np.ndarray, pd.DataFrame, pd.Series)) or \
                not isinstance(target, (np.ndarray, pd.DataFrame, pd.Series)):
            msg = f"{ErrorNumbers.FB609.value}. Parameters `inputs` and `target` for " \
                  f"initialization of {self.__class__.__name__} should be one of np.ndarray, pd.DataFrame, pd.Series"
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        # Convert pd.DataFrame or pd.Series to np.ndarray for `inputs`
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            self._inputs = inputs.to_numpy()
        else:
            self._inputs = inputs

        # Convert pd.DataFrame or pd.Series to np.ndarray for `target`
        if isinstance(target, (pd.DataFrame, pd.Series)):
            self._target = target.to_numpy()
        else:
            self._target = target

        # Additional loader arguments
        self._loader_arguments = kwargs

        # Subset None means that train/validation split has not been performed
        self._subset_test: Union[Tuple[np.ndarray, np.ndarray], None] = None
        self._subset_train: Union[Tuple[np.ndarray, np.ndarray], None] = None

    def dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the entire registered dataset.

        This method returns whole dataset as it is without any split.

        Returns:
             inputs: Input variables for model training
             targets: Target variable for model training
        """
        return self._inputs, self._target

    def subset_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets Subset of dataset for validation partition.

        Returns:
            test_inputs: Input variables of validation subset for model validation
            test_target: Target variable of validation subset for model validation
        """
        return self._subset_test

    def subset_train(self) -> Tuple[np.ndarray, np.ndarray]:

        """Gets Subset for train partition.

        Returns:
            test_inputs: Input variables of training subset for model training
            test_target: Target variable of training subset for model training
        """

        return self._subset_train

    def split(self, test_ratio: float) -> Tuple[Optional[NPDataLoader], Optional[NPDataLoader]]:
        """Splits `np.ndarray` dataset into train and validation.

        Args:
             test_ratio: Ratio for validation set partition. Rest of the samples will be used for training

        Raises:
            FedbiomedSkLearnDataManagerError: If the `test_ratio` is not between 0 and 1

        Returns:
             train_loader: NPDataLoader of input variables for model training
             test_loader: NPDataLoader of target variable for model training
        """
        if not isinstance(test_ratio, float):
            msg = f'{ErrorNumbers.FB609.value}: The argument `ratio` should be type `float` not {type(test_ratio)}'
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        if test_ratio < 0. or test_ratio > 1.:
            msg = f'{ErrorNumbers.FB609.value}: The argument `ratio` should be equal or between 0 and 1, ' \
                 f'not {test_ratio}'
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        empty_subset = (np.array([]), np.array([]))

        if test_ratio <= 0.:
            self._subset_train = (self._inputs, self._target)
            self._subset_test = empty_subset
        elif test_ratio >= 1.:
            self._subset_train = empty_subset
            self._subset_test = (self._inputs, self._target)
        else:
            x_train, x_test, y_train, y_test = train_test_split(self._inputs, self._target, test_size=test_ratio)
            self._subset_test = (x_test, y_test)
            self._subset_train = (x_train, y_train)

        return self._subset_loader(self._subset_train, **self._loader_arguments), \
            self._subset_loader(self._subset_test, **self._loader_arguments)

    @staticmethod
    def _subset_loader(subset: Tuple[np.ndarray, np.ndarray], **loader_arguments) -> Optional[NPDataLoader]:
        """Loads subset partition for SkLearn based training plans.

        Raises:
            FedbiomedSkLearnDataManagerError: If subset is not well formatted

        Returns:
            A NPDataLoader encapsulating the subset
        """
        if not isinstance(subset, Tuple) \
                or len(subset) != 2 \
                or not isinstance(subset[0], np.ndarray) \
                or not isinstance(subset[1], np.ndarray):

            raise FedbiomedTypeError(f'{ErrorNumbers.FB609.value}: The argument `subset` should a Tuple of size 2 '
                                     f'that contains inputs/data and target as np.ndarray.')

        # Empty validation set
        if len(subset[0]) == 0 or len(subset[1]) == 0:
            return None

        try:
            loader = NPDataLoader(dataset=subset[0], target=subset[1], **loader_arguments)
        except TypeError:
            valid_loader_arguments = get_method_spec(NPDataLoader)
            valid_loader_arguments.pop('dataset')
            valid_loader_arguments.pop('target')
            msg = f"{ErrorNumbers.FB609.value}. Wrong keyword loader arguments for NPDataLoader. Valid arguments " \
                  f"are: {[k for k in valid_loader_arguments.keys()]}, instead got {[k for k in loader_arguments]}."
            logger.error(msg)
            raise FedbiomedTypeError(msg)

        return loader

