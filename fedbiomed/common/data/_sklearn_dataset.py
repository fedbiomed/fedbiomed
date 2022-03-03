import numpy as np
import pandas as pd
from typing import Union, Tuple

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from fedbiomed.common.exceptions import FedbiomedError


class SkLearnDataset(object):

    def __init__(self,
                 inputs: Union[np.ndarray, pd.DataFrame, pd.Series],
                 target: Union[np.ndarray, pd.DataFrame, pd.Series],
                 **kwargs):

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
        self._subset_test: Tuple[np.ndarray, np.ndarray, None] = None
        self._subset_train: Tuple[np.ndarray, np.ndarray, None] = None

    def dataset(self) -> Tuple[Union[ndarray, DataFrame, Series],
                               Union[ndarray, DataFrame, Series]]:
        """
        Getter for dataset.

        Returns:
            torch.utils.data.Dataset
        """
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

    def load_test_partition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for loading testing partition of Dataset as pytorch DataLoader. Before calling
        this method Dataset should be split into test and train subset in advance

        Raises:
            FedbiomedError: If Dataset is not split into test and train in advance
        """
        if self._subset_test is None:
            raise FedbiomedError()
        # TODO: Create DataLoader for SkLearnDataset to apply batch training
        return self._subset_test

    def load_train_partition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for loading training partition of Dataset as SkLearnDataset. Before calling
        this method Dataset should be split into test and train subset in advance

        Raises:
            FedbiomedError: If Dataset is not split into test and train in advance
        """
        if self._subset_train is None:
            raise FedbiomedError()

        # TODO: Create DataLoader for SkLearnDataset to apply batch training
        return self._subset_train

    def split(self, ratio: float) -> None:

        # Check ratio is valid for splitting
        if ratio < 0 or ratio > 1:
            raise FedbiomedError('The argument `ratio` should be between 0 and 1, not {ratio}')

        x_train, x_test, y_train, y_test = train_test_split(self._inputs, self._target, test_size=ratio)

        self._subset_test = (x_test, y_test)
        self._subset_train = (x_train, y_train)

        return None
