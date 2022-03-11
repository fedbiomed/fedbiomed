import math

from typing import Union, Tuple
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data import random_split
from fedbiomed.common.exceptions import FedbiomedTorchDataManagerError
from fedbiomed.common.constants import ErrorNumbers
from ._sklearn_data_manager import SkLearnDataManager


class TorchDataManager(object):

    def __init__(self, dataset: Union[Dataset], **kwargs):
        """
        Wrapper for torch.utils.data.Dataset to manager loading operations for
        test and train, it is a manager for Torch based Dataset object.

        Args:
            dataset (Dataset, MonaiDataset): Dataset object for torch.utils.data.DataLoader

        Attr:
            _loader_arguments: The arguments that are going to be passed to torch.utils.data.DataLoader
            _subset_test: Test subset of dataset
            _subset_train: Train subset of dataset

        Raises:
            FedbiomedTorchDataManagerError: If the argument `dataset` is not an instance of `torch.utils.data.Dataset`
        """

        # TorchDataManager should get `dataset` argument as an instance of torch.utils.data.Dataset
        if not isinstance(dataset, Dataset):
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: The attribute `dataset` should an instance "
                f"of `torch.utils.data.Dataset`, please use `Dataset` as parent class for"
                f"your custom torch dataset object")

        self._dataset = dataset
        self._loader_arguments = kwargs
        self._subset_test: Union[Subset, None] = None
        self._subset_train: Union[Subset, None] = None

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

    def load_all_samples(self) -> DataLoader:
        """
        Method for loading all samples as PyTorch DataLoader without splitting. If researcher
        requests training without testing this method can be used
        """

        try:
            loader = DataLoader(self._dataset, **self._loader_arguments)
        except TypeError as err:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Error while creating a PyTorch DataLoader "
                f"for all samples due to loader arguments: {str(err)}")

        return loader

    def split(self, test_ratio: float) -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
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
        if not isinstance(test_ratio, (float, int)):
            raise FedbiomedTorchDataManagerError(f'{ErrorNumbers.FB608.value}: The argument `ratio` should be '
                                                 f'type `float` or `int` not {type(test_ratio)}')

        # Check ratio is valid for splitting
        if test_ratio < 0 or test_ratio > 1:
            raise FedbiomedTorchDataManagerError(f'{ErrorNumbers.FB608.value}: The argument `ratio` should be '
                                                 f'equal or between 0 and 1, not {test_ratio}')

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
        test_samples = math.floor(samples * test_ratio)
        train_samples = samples - test_samples

        self._subset_train, self._subset_test = random_split(self._dataset, [train_samples, test_samples])

        loaders = (self._subset_loader(self._subset_train, **self._loader_arguments),
                   self._subset_loader(self._subset_test, batch_size=len(self._subset_test)))

        return loaders

    def to_sklearn(self):
        """
         Method to convert `torch.utils.data.Dataset` dataset to numpy based object
         and create a SkLearnDataManager to use in SkLearnTraining base training plans
        """

        # Create a loader from self._dataset to extract inputs and target values
        # by iterating over samples
        loader = DataLoader(self._dataset, batch_size=len(self._dataset))

        # Iterate over samples and get input variable and target variable
        inputs = next(iter(loader))[0].numpy()
        target = next(iter(loader))[1].numpy()

        return SkLearnDataManager(inputs=inputs, target=target)

    @staticmethod
    def _subset_loader(subset: Subset, **kwargs) -> DataLoader:
        """
        Method for loading subset (train/test) partition of as pytorch DataLoader.

        Args:
            subset (Subset): Subset as an instance of PyTorch's `Subset`
        Raises:
            FedbiomedError: If Dataset is not split into test and train in advance
        """

        # Return None if subset has no data
        if len(subset) <= 0:
            return None

        try:
            loader = DataLoader(subset, **kwargs)
        except TypeError as err:
            raise FedbiomedTorchDataManagerError(
                f"{ErrorNumbers.FB608.value}: Error while creating a PyTorch DataLoader "
                f"due to loader arguments: {str(err)}")

        return loader
