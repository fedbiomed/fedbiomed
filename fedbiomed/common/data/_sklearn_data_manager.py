# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data Management for scikit-learn in Fed-BioMed.

This module introduces the NPDataLoader and SkLearnDataManager classes, to provide a data management interface for
Fed-BioMed users relying on the scikit-learn framework that is similar to the interface for torch.
"""


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import (
    FedbiomedError,
    FedbiomedTypeError,
    FedbiomedValueError,
)
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


class SkLearnDataManager(object):
    """Wrapper for `pd.DataFrame`, `pd.Series` and `np.ndarray` datasets.

    Manages datasets for scikit-learn based model training. Responsible for managing inputs, and target
    variables that have been provided in `training_data` of scikit-learn based training plans.
    """

    def __init__(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, pd.Series],
        target: Union[np.ndarray, pd.DataFrame, pd.Series],
        **kwargs: dict,
    ):
        """Construct a SkLearnDataManager from an array of inputs and an array of targets.

        The loader arguments will be passed to the [fedbiomed.common.data.NPDataLoader] classes instantiated
        when split is called. They may include batch_size, shuffle, drop_last, and others. Please see the
        [fedbiomed.common.data.NPDataLoader] class for more details.

        Args:
            inputs: Independent variables (inputs, features) for model training
            target: Dependent variable/s (target) for model training and validation
            **kwargs: Loader arguments
        """

        if not isinstance(
            inputs, (np.ndarray, pd.DataFrame, pd.Series)
        ) or not isinstance(target, (np.ndarray, pd.DataFrame, pd.Series)):
            msg = (
                f"{ErrorNumbers.FB609.value}. Parameters `inputs` and `target` for "
                f"initialization of {self.__class__.__name__} should be one of np.ndarray, pd.DataFrame, pd.Series"
            )
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

        # rand_seed = kwargs.get('random_seed')
        # self.rng(rand_seed)

        # Subset None means that train/validation split has not been performed
        self._subset_test: Union[Tuple[np.ndarray, np.ndarray], None] = None
        self._subset_train: Union[Tuple[np.ndarray, np.ndarray], None] = None

        self.training_index: List[int] = []
        self.testing_index: List[int] = []
        self.test_ratio: Optional[float] = None
        self._is_shuffled_testing_dataset: bool = False
        if "shuffle_testing_dataset" in kwargs:
            self._is_shuffled_testing_dataset: bool = kwargs.pop(
                "shuffle_testing_dataset"
            )

        # Additional loader arguments
        self._loader_arguments = kwargs

    def dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the entire registered dataset.

        This method returns whole dataset as it is without any split.

        Returns:
             inputs: Input variables for model training
             targets: Target variable for model training
        """
        return self._inputs, self._target

    def subset_test(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Gets Subset of dataset for validation partition.

        Returns:
            test_inputs: Input variables of validation subset for model validation
            test_target: Target variable of validation subset for model validation
        """
        return self._subset_test

    def subset_train(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Gets Subset for train partition.

        Returns:
            test_inputs: Input variables of training subset for model training
            test_target: Target variable of training subset for model training
        """

        return self._subset_train

    def split(
        self,
        test_ratio: float,
        test_batch_size: int,
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[NPDataLoader, NPDataLoader]:
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
            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB609.value}: The argument `ratio` should be type "
                f"`float` not {type(test_ratio)}"
            )

        if test_ratio < 0.0 or test_ratio > 1.0:
            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB609.value}: The argument `ratio` should be equal or between "
                f"0 and 1, not {test_ratio}"
            )

        empty_subset = (np.array([]), np.array([]))

        if self.test_ratio != test_ratio and self.test_ratio is not None:
            if not is_shuffled_testing_dataset:
                logger.info(
                    "`test_ratio` value has changed: this will change the testing dataset"
                )
            is_shuffled_testing_dataset = True

        if test_ratio <= 0.0:
            self._subset_train = (self._inputs, self._target)
            self._subset_test = empty_subset
            self.training_index, self.testing_index = list(range(len(self._inputs))), []
        elif test_ratio >= 1.0:
            self._subset_train = empty_subset
            self._subset_test = (self._inputs, self._target)
            self.training_index, self.testing_index = [], list(range(len(self._inputs)))

        else:
            _is_loading_failed: bool = False
            if self.testing_index and not is_shuffled_testing_dataset:
                # reloading testing dataset from previous rounds
                try:
                    self._load_indexes(self.training_index, self.testing_index)
                except IndexError:
                    _is_loading_failed = True
            if (
                not self.testing_index
                or is_shuffled_testing_dataset
                or _is_loading_failed
            ):
                (x_train, x_test, y_train, y_test, idx_train, idx_test) = (
                    train_test_split(
                        self._inputs,
                        self._target,
                        np.arange(len(self._inputs)),
                        test_size=test_ratio,
                    )
                )
                self._subset_test = (x_test, y_test)
                self._subset_train = (x_train, y_train)
                self.training_index = idx_train.tolist()
                self.testing_index = idx_test.tolist()

        if not test_batch_size:
            test_batch_size = len(self._subset_test)

        self.test_ratio = test_ratio  # float(np.clip(0, 1, test_ratio))

        # self._loader_arguments['random_seed'] = self._rng
        return self._subset_loader(
            self._subset_train, **self._loader_arguments
        ), self._subset_loader(self._subset_test, batch_size=test_batch_size)

    def save_state(self) -> Dict:
        """Gets state of the data loader.

        Returns:
            A Dict containing data loader state.
        """
        _loader_args = {}
        _loader_args["training_index"], _loader_args["testing_index"] = (
            self.training_index,
            self.testing_index,
        )
        _loader_args["test_ratio"] = self.test_ratio

        return _loader_args

    def load_state(self, state: Dict) -> None:
        """Loads state of the data loader


        It currently keep only testing index, training index and test ratio
        as state.

        Args:
            state: Object containing data loader state.
        """

        self.testing_index = state.get("testing_index", [])
        self.training_index = state.get("training_index", [])
        self.test_ratio = state.get("test_ratio", None)

    def _load_indexes(self, training_idx: List[int], testing_idx: List[int]):

        try:
            self._subset_train = (
                self._inputs[training_idx],
                self._target[training_idx],
            )
            self._subset_test = (self._inputs[testing_idx], self._target[testing_idx])
        except IndexError as e:
            raise FedbiomedError(
                f"Cannot load testing dataset, probably because dataset have changed.\n"
                f"Hence, dataset will be reshuffled. More details: {e}"
            ) from e

    @staticmethod
    def _subset_loader(
        subset: Tuple[np.ndarray, np.ndarray], **loader_arguments
    ) -> Optional[NPDataLoader]:
        """Loads subset partition for SkLearn based training plans.

        Raises:
            FedbiomedSkLearnDataManagerError: If subset is not well formatted

        Returns:
            A NPDataLoader encapsulating the subset
        """
        if (
            not isinstance(subset, Tuple)
            or len(subset) != 2
            or not isinstance(subset[0], np.ndarray)
            or not isinstance(subset[1], np.ndarray)
        ):
            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB609.value}: The argument `subset` should a Tuple of size 2 "
                f"that contains inputs/data and target as np.ndarray."
            )
        try:
            loader = NPDataLoader(
                dataset=subset[0], target=subset[1], **loader_arguments
            )
        except TypeError as e:
            valid_loader_arguments = get_method_spec(NPDataLoader)
            valid_loader_arguments.pop("dataset")
            valid_loader_arguments.pop("target")
            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB609.value}. Wrong keyword loader arguments for NPDataLoader. "
                f"Full error message was: {e} Valid arguments are: "
                f"{[k for k in valid_loader_arguments.keys()]}, "
                f"instead got {[k for k in loader_arguments]}. "
            ) from e
        except ValueError as e:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB609.value}. Wrong value of loader arguments"
                f"for NPDataLoader. Full error message was: {e}"
            ) from e

        return loader
