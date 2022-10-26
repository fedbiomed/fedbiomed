"""
Data Management classes
"""

from typing import Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from fedbiomed.common.constants import _BaseEnum, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedDataManagerError

from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import NPDataLoader, SkLearnDataManager
from ._tabular_dataset import TabularDataset


class DataLoaderTypes(_BaseEnum):
    """Enum for data-loader classes."""

    NUMPY = NPDataLoader
    TORCH = DataLoader


# Type hint alias for Union of DataLoaderTypes concrete classes.
TypeDataLoader = Union[
    NPDataLoader,
    DataLoader,
]


class DataManager:
    """Factory class that build different data loader/datasets based on the type of `dataset`.
    The argument `dataset` should be provided as `torch.utils.data.Dataset` object for to be used in
    PyTorch training.
    """
    def __init__(self,
                 dataset: Union[np.ndarray, pd.DataFrame, pd.Series, Dataset],
                 target: Union[np.ndarray, pd.DataFrame, pd.Series] = None,
                 **kwargs: dict) -> None:

        """Constructor of DataManager,

        Args:
            dataset: Dataset object. It can be an instance, PyTorch Dataset or Tuple.
            target: Target variable or variables.
            **kwargs: Additional parameters that are going to be used for data loader
        """

        # TODO: Improve datamanager for auto loading by given dataset_path and other information
        # such as inputs variable indexes and target variables indexes

        self._dataset = dataset
        self._target = target
        self._loader_arguments = kwargs
        self._data_manager_instance = None

    def load(self, loader_type: DataLoaderTypes):
        """Loads proper DataManager based on given TrainingPlan and
        `dataset`, `target` attributes.

        Args:
            loader_type: Type of dataloader (torch DataLoader, NPDataLoader...)
                required by the TrainingPlan.

        Raises:
            FedbiomedDataManagerError: If requested DataManager does not match with given arguments.

        """

        # Training plan is type of TorcTrainingPlan
        if loader_type is DataLoaderTypes.TORCH:
            if self._target is None and isinstance(self._dataset, Dataset):
                # Create Dataset for pytorch
                self._data_manager_instance = TorchDataManager(dataset=self._dataset, **self._loader_arguments)
            elif isinstance(self._dataset, (pd.DataFrame, pd.Series, np.ndarray)) and \
                    isinstance(self._target, (pd.DataFrame, pd.Series, np.ndarray)):
                # If `dataset` and `target` attributes are array-like object
                # create TabularDataset object to instantiate a TorchDataManager
                torch_dataset = TabularDataset(inputs=self._dataset, target=self._target)
                self._data_manager_instance = TorchDataManager(dataset=torch_dataset, **self._loader_arguments)
            else:
                raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Invalid arguments for torch based "
                                                f"training plan, either provide the argument  `dataset` as PyTorch "
                                                f"Dataset instance, or provide `dataset` and `target` arguments as "
                                                f"an instance one of pd.DataFrame, pd.Series or np.ndarray ")

        elif loader_type is DataLoaderTypes.NUMPY:
            # Try to convert `torch.utils.Data.Dataset` to SkLearnBased dataset/datamanager
            if self._target is None and isinstance(self._dataset, Dataset):
                torch_data_manager = TorchDataManager(dataset=self._dataset)
                try:
                    self._data_manager_instance = torch_data_manager.to_sklearn()
                except Exception as e:
                    raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: PyTorch based `Dataset` object "
                                                    "has been instantiated with DataManager. An error occurred while"
                                                    "trying to convert torch.utils.data.Dataset to numpy based "
                                                    f"dataset: {str(e)}")

            # For scikit-learn based training plans, the arguments `dataset` and `target` should be an instance
            # one of `pd.DataFrame`, `pd.Series`, `np.ndarray`
            elif isinstance(self._dataset, (pd.DataFrame, pd.Series, np.ndarray)) and \
                    isinstance(self._target, (pd.DataFrame, pd.Series, np.ndarray)):
                # Create Dataset for SkLearn training plans
                self._data_manager_instance = SkLearnDataManager(inputs=self._dataset, target=self._target,
                                                                 **self._loader_arguments)
            else:
                raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: The argument `dataset` and `target` "
                                                f"should be instance of pd.DataFrame, pd.Series or np.ndarray ")
        else:
            raise FedbiomedDataManagerError(
                f"{ErrorNumbers.FB607.value}: Unsupported data loader type: {loader_type}"
            )

    def __getattr__(self, item: str):

        """Wraps all functions/attributes of factory class members.

        Args:
             item: Requested item from class

        Raises:
            FedbiomedDataManagerError: If the attribute is not implemented

        """

        # Specific to DataManager class
        if item == 'load':
            return object.__getattribute__(self, item)
        try:
            return self._data_manager_instance.__getattribute__(item)
        except AttributeError:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: method {str(item)} not "
                                            f"implemented for class: {str(self._data_manager_instance)}")
