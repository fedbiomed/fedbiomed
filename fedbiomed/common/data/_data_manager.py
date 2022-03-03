import inspect
import numpy as np
import pandas as pd
from typing import Union, Tuple, Callable
from torch.utils.data import Dataset
from fedbiomed.common.exceptions import FedbiomedError
from ._torch_dataset import TorchDataset
from ._sklearn_dataset import SkLearnDataset


class DataManager(object):

    def __init__(self,
                 dataset: Union[np.ndarray, pd.DataFrame, pd.Series, Dataset] = None,
                 target: Union[np.ndarray, pd.DataFrame, pd.Series] = None,
                 **kwargs):

        """
        Constructor of DataManager, it is factory class that build different data loader/datasets
        based on the type of `dataset`

        Args:
            - dataset (MonaiDataset, Tuple, Dataset): Dataset object. It can be an instance of monai Dataset,
            PyTorch Dataset or Tuple.

            - **kwargs: Additional parameters that are going to be used for data loader
        """

        if target is None and isinstance(dataset, Dataset):
            # Create Dataset for pytorch
            self.__dataset = TorchDataset(dataset=dataset, **kwargs)
        else:
            # If target is None `inputs` should be an instance of np.ndarray
            # pd.DataFrame or pd.Series
            if isinstance(dataset, Dataset):
                raise FedbiomedError(f"")
            elif isinstance(dataset, (pd.DataFrame, pd.Series, np.ndarray)) and \
                    isinstance(target, (pd.DataFrame, pd.Series, np.ndarray)):
                # Create Dataset for SkLearn training plans
                self.__dataset = SkLearnDataset(inputs=dataset, target=target, **kwargs)
            else:
                raise FedbiomedError

    def __getattr__(self, item):

        """
        Wrap all functions/attributes of factory class members.

        Args:
             item: Requested item from class

        Raises:
            FedbiomedDataManagerError: If the attribute is not implemented

        """
        try:
            return self.__dataset.__getattribute__(item)
        except AttributeError:
            raise FedbiomedError(f"method {str(item)} not implemented for class: " + str(self._datatype))

