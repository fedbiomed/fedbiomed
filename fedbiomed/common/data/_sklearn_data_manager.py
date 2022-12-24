# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

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
from fedbiomed.common.data.loaders import NPDataLoader


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

    def split(self, test_ratio: float) -> Tuple[NPDataLoader, NPDataLoader]:
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

        test_batch_size = max(1, len(self._subset_test[0]))
        return self._subset_loader(self._subset_train, **self._loader_arguments), \
            self._subset_loader(self._subset_test, batch_size=test_batch_size)

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

        try:
            loader = NPDataLoader(dataset=subset[0], target=subset[1], **loader_arguments)
        except TypeError as e:
            valid_loader_arguments = get_method_spec(NPDataLoader)
            valid_loader_arguments.pop('dataset')
            valid_loader_arguments.pop('target')
            msg = f"{ErrorNumbers.FB609.value}. Wrong keyword loader arguments for NPDataLoader. " \
                  f"Full error message was: {e}" \
                  f"Valid arguments are: {[k for k in valid_loader_arguments.keys()]}, " \
                  f"instead got {[k for k in loader_arguments]}. "
            logger.error(msg)
            raise FedbiomedTypeError(msg)
        except ValueError as e:
            msg = f"{ErrorNumbers.FB609.value}. Wrong value of loader arguments for NPDataLoader. " \
                  f"Full error message was: {e}"
            logger.error(msg)
            raise FedbiomedValueError(msg)

        return loader

