import math
import inspect

from typing import Union
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data import random_split
from fedbiomed.common.exceptions import FedbiomedTorchDataManagerError
from fedbiomed.common.constants import ErrorNumbers


class TorchDataManager(object):

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
            FedbiomedTorchDataManagerError: If the argument `dataset` is not an instance of `torch.utils.data.Dataset`
        """

        if not isinstance(dataset, Dataset):
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: The attribute `dataset` should an instance "
                f"of `torch.utils.data.Dataset`, please use `Dataset` as parent class for"
                f"your custom torch dataset object")

        self._dataset = dataset
        self._loader_arguments = kwargs
        self._subset_test: Union[Subset, None] = None
        self._subset_train: Union[Subset, None] = None

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
            raise FedbiomedTorchDataManagerError(f"{ErrorNumbers.FB608.value}: Can not find subset for test partition. "
                                                 f"Please make sure that the method `.split(ratio=ration)` DataManager "
                                                 f"object has been called before. ")
        # No test loader
        if len(self._subset_test) <= 0:
            return None

        try:
            loader = DataLoader(self._subset_test, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Error while creating a PyTorch DataLoader "
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
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Can not find subset for train partition. "
                f"Please make sure that the method `.split(ratio=ration)` DataManager "
                f"object has been called before. ")
        # No train subset
        if len(self._subset_train) <= 0:
            return None

        try:
            loader = DataLoader(self._subset_train, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Error while creating a PyTorch DataLoader "
                f"for train partition due to loader arguments: {str(err)}")

        return loader

    def load_all_samples(self) -> DataLoader:
        """
        Method for loading all samples as PyTorch DataLoader without splitting
        """

        try:
            loader = DataLoader(self._dataset, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Error while creating a PyTorch DataLoader "
                f"for all samples due to loader arguments: {str(err)}")

        return loader

    def split(self, ratio: float) -> None:
        """
        Method for splitting PyTorch Dataset into train and test.

        Args:
             ratio (float): Split ratio for testing set ratio. Rest of the samples
                            will be used for training
        Raises:
            FedbiomedTorchDataManagerError: If the ratio is not in good format

        Returns:
             none
        """

        # Check the argument `ratio` is of type `float`
        if not isinstance(ratio, (float, int)):
            raise FedbiomedTorchDataManagerError(f'{ErrorNumbers.FB608.value}: The argument `ratio` should be '
                                                 f'type `float` or `int` not {type(ratio)}')

        # Check ratio is valid for splitting
        if ratio < 0 or ratio > 1:
            raise FedbiomedTorchDataManagerError(f'{ErrorNumbers.FB608.value}: The argument `ratio` should be '
                                                 f'equal or between 0 and 1, not {ratio}')

        # If `Dataset` has proper data attribute
        # try to get shape from self.data
        if not hasattr(self._dataset, '__len__'):
            raise FedbiomedTorchDataManagerError(f"{ErrorNumbers.FB608.value}: Can not get number of samples from "
                                                 f"{str(self._dataset)} without `__len__`.  Please make sure "
                                                 f"that `__len__` method has been added to custom dataset. "
                                                 f"This method should return total number of samples.")

        try:
            samples = len(self._dataset)
        except AttributeError as e:
            raise FedbiomedTorchDataManagerError(f"{ErrorNumbers.FB608.value}: Can not get number of samples from "
                                                 f"{str(self._dataset)}, {str(e)}")
        except TypeError as e:
            raise FedbiomedTorchDataManagerError(f"{ErrorNumbers.FB608.value}: Can not get number of samples from "
                                                 f"{str(self._dataset)}, {str(e)}")

        # Calculate number of samples for train and test subsets
        test_samples = math.floor(samples * ratio)
        train_samples = samples - test_samples

        self._subset_train, self._subset_test = random_split(self._dataset, [train_samples, test_samples])
