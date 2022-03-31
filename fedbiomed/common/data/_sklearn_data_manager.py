"""
Sklearn data manager
"""


from typing import Union, Tuple

import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame, Series

from sklearn.model_selection import train_test_split
from fedbiomed.common.exceptions import FedbiomedSkLearnDataManagerError
from fedbiomed.common.constants import ErrorNumbers


class SkLearnDataManager(object):

    def __init__(self,
                 inputs: Union[np.ndarray, pd.DataFrame, pd.Series],
                 target: Union[np.ndarray, pd.DataFrame, pd.Series],
                 **kwargs):

        """
        Wrapper for `pd.DataFrame`, `pd.Series` and `np.ndarray` datasets that is going to  be
        used for scikit-learn based model training. This class is responsible for managing inputs, and
        target variables that have been provided in `training_data` of scikit-learn based training
        plans.

        Args:
            inputs (np.ndarray, pd.DataFrame, pd.Series): Independent variables (inputs, features) for model training
            target (np.ndarray, pd.DataFrame, pd.Series): Dependent variable/s (target) for model training and
                                                            evaluation

        Attr:
            _loader_arguments: The arguments that are going to be passed to torch.utils.data.DataLoader
            _subset_test: Test subset of dataset
            _subset_train: Train subset of dataset

        Raises:
            none
        """

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

        # Subset None means that train/test split has not been performed
        self._subset_test: Union[Tuple[np.ndarray, np.ndarray], None] = None
        self._subset_train: Union[Tuple[np.ndarray, np.ndarray], None] = None

    def dataset(self) -> Tuple[Union[ndarray, DataFrame, Series],
                               Union[ndarray, DataFrame, Series]]:
        """
        Getter for dataset. This returns whole dataset as it is without any split.

        Returns:
             Tuple[Union[ndarray, DataFrame, Series], Union[ndarray, DataFrame, Series]]
        """

        # TODO: When a proper DataLoader is develop for SkLearn framework, this method should
        # return pure data not data loader.  The method load_all_samples() should return dataloader
        # please see the method load_all_samples()

        return self._inputs, self._target

    def subset_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Getter for Subset of dataset for test partition.

        Raises:
            none

        Returns:
            torch.utils.data.Subset | None
        """

        return self._subset_test

    def subset_train(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        Getter for Subset for train partition.

        Raises:
            none

        Returns:
            torch.utils.data.Subset | None
        """

        return self._subset_train

    def load_all_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for loading all samples as Numpy ndarray without splitting
        """

        # TODO: Return batch iterator
        return self._inputs, self._target

    def split(self, test_ratio: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Method for splitting np.ndarray dataset into train and test.

        Args:
             test_ratio (float): Ratio for testing set partition. Rest of the samples
                            will be used for training
        Raises:
            FedbiomedSkLearnDataManagerError

        Returns:
             none
        """

        # Check the argument `ratio` is of type `float`
        if not isinstance(test_ratio, (float, int)):
            raise FedbiomedSkLearnDataManagerError(f'{ErrorNumbers.FB608.value}: The argument `ratio` should be '
                                                   f'type `float` or `int` not {type(test_ratio)}')

        # Check ratio is valid for splitting
        if test_ratio < 0 or test_ratio > 1:
            raise FedbiomedSkLearnDataManagerError(f'{ErrorNumbers.FB609.value}: The argument `ratio` should be '
                                                   f'equal or between 0 and 1, not {test_ratio}')

        empty_subset = (np.array([]), np.array([]))

        if test_ratio == 0:
            self._subset_train = (self._inputs, self._target)
            self._subset_test = empty_subset
        elif test_ratio == 1:
            self._subset_train = empty_subset
            self._subset_test = (self._inputs, self._target)
        else:
            x_train, x_test, y_train, y_test = train_test_split(self._inputs, self._target, test_size=test_ratio)
            self._subset_test = (x_test, y_test)
            self._subset_train = (x_train, y_train)

        return self._subset_loader(self._subset_train), self._subset_loader(self._subset_test)

    @staticmethod
    def _subset_loader(subset: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for loading subset partition for SkLearn based training plans.

        TODO: Currently this method just returns subset. When SkLearn based batch
        iterator is created, it should return BatchIterator
        """
        if not isinstance(subset, Tuple) \
                or len(subset) != 2 \
                or not isinstance(subset[0], np.ndarray) \
                or not isinstance(subset[1], np.ndarray):

            raise FedbiomedSkLearnDataManagerError(f'{ErrorNumbers.FB609.value}: The argument `subset` should a Tuple'
                                                   f'of size 2 that contains inputs/data and target as np.ndarray.')

        # Empty test set
        if len(subset[0]) == 0 or len(subset[0]) == 0:
            return None

        # TODO: Return DataLoader/BatchIterator for SkLearnDataset to apply batch training
        # Example:
        # return BatchIterator(subset, **self.loader_arguments)

        return subset
