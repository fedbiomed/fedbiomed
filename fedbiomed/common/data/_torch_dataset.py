import math
import inspect

from typing import Union
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data import random_split
from fedbiomed.common.exceptions import FedbiomedDataManagerError
from fedbiomed.common.constants import ErrorNumbers


class TorchDataset(object):

    def __init__(self, dataset: Union[Dataset], **kwargs):
        """
        Wrapper for torch.utils.data.Dataset

        Args:
            dataset (Dataset, MonaiDataset): Dataset object for torch.utils.data.DataLoader

        Attr:
            _loader_arguments: The arguments that are going to be passed to torch.utils.data.DataLoader
            _subset_test: Test subset of dataset
            _subset_train: Train subset of dataset

        Raises:
            none
        """

        self._dataset = dataset
        self._loader_arguments = kwargs
        self._subset_test = None
        self._subset_train = None

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)

    def dataset(self) -> Dataset:
        """
        Getter for dataset.

        Returns:
            torch.utils.data.Dataset
        """
        return self._dataset

    def subset_test(self) -> Subset:
        """
        Getter for Subset of dataset for test partition.

        Raises:
            none

        Returns:
            torch.utils.data.Subset | None
        """

        return self._subset_test

    def subset_train(self) -> Subset:

        """
        Getter for Subset for train partition.

        Raises:
            none

        Returns:
            torch.utils.data.Subset | None
        """

        return self._subset_train

    def load_test_partition(self) -> DataLoader:
        """
        Method for loading testing partition of Dataset as pytorch DataLoader. Before calling
        this method Dataset should be split into test and train subset in advance

        Raises:
            FedbiomedError: If Dataset is not split into test and train in advance
        """

        if self._subset_test is None:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Can not find subset for test partition. "
                                            f"Please make sure that the method `.split(ratio=ration)` DataManager "
                                            f"object has been called before. ")

        try:
            loader = DataLoader(self._subset_test, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Error while creating a PyTorch DataLoader "
                                            f"for test partition due to loader arguments: {str(err)}")

        return loader

    def load_train_partition(self) -> DataLoader:
        """
        Method for loading training partition of Dataset as pytorch DataLoader. Before calling
        this method Dataset should be split into test and train subset in advance

        Raises:
            FedbiomedError: If Dataset is not split into test and train in advance
        """
        if self._subset_train is None:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Can not find subset for train partition. "
                                            f"Please make sure that the method `.split(ratio=ration)` DataManager "
                                            f"object has been called before. ")

        try:
            loader = DataLoader(self._subset_train, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Error while creating a PyTorch DataLoader "
                                            f"for train partition due to loader arguments: {str(err)}")

        return loader

    def load_all_samples(self) -> DataLoader:
        """
        Method for loading all samples as PyTorch DataLoader without splitting
        """

        try:
            loader = DataLoader(self._dataset, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Error while creating a PyTorch DataLoader "
                                            f"for all samples due to loader arguments: {str(err)}")

        return loader

    def split(self, ratio: float) -> None:
        """
        Method for splitting PyTorch Dataset into train and test.

        Args:
             ratio (float): Split ratio for testing set ratio. Rest of the samples
                            will be used for training
        Raises:
            FedbiomedDataManagerError: If the ratio is not in good format

        Returns:
             none
        """
        # Check ratio is valid for splitting
        if ratio < 0 or ratio > 1:
            raise FedbiomedDataManagerError(f'{ErrorNumbers.FB607.value}: The argument `ratio` should be '
                                            f'equal or between 0 and 1, not {ratio}')

        # Get number of samples of dataset
        samples = self._dataset.data.shape[0]

        # Calculate number of samples for train and test subsets
        test_samples = math.floor(samples * ratio)
        train_samples = samples - test_samples

        self._subset_train, self._subset_test = random_split(self._dataset, [train_samples, test_samples])
