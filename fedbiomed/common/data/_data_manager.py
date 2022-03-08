import numpy as np
import pandas as pd

from typing import Union, Tuple, Callable
from torch.utils.data import Dataset
from fedbiomed.common.exceptions import FedbiomedDataManagerError
from fedbiomed.common.constants import ErrorNumbers, TrainingPlans

from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import SkLearnDataManager
from ._torch_tabular_dataset import TorchTabularDataset


class DataManager(object):

    def __init__(self,
                 dataset: Union[np.ndarray, pd.DataFrame, pd.Series, Dataset],
                 target: Union[np.ndarray, pd.DataFrame, pd.Series] = None,
                 **kwargs) -> None:

        """
        Constructor of DataManager, it is factory class that build different data loader/datasets
        based on the type of `dataset`. The argument `dataset` should be provided as `torch.utils.data.Dataset`
        object for to be used in PyTorch training. Otherwise,

        Args:
            - dataset (MonaiDataset, Tuple, Dataset): Dataset object. It can be an instance,
            PyTorch Dataset or Tuple.

            - **kwargs: Additional parameters that are going to be used for data loader


        """

        # TODO: Improve datamanager for auto loading by given dataset_path and other information
        # such as inputs variable indexes and target variables indexes

        self.dataset = dataset
        self.target = target
        self.loader_arguments = kwargs

    def load(self, tp_type: TrainingPlans):
        """
        Method for loading proper DataManager based on given TrainingPlan and
        `dataset`, `target` attributes.

        Args:
            tp_type (TrainingPlans): Enumeration instance of TrainingPlans that stands for
                                     type of training plan.

        Raises:

        - FedbiomedDataManagerError: - If requested DataManager does not match with given
                                        arguments.

        """

        # Training plan is type of TorcTrainingPlan
        if tp_type == TrainingPlans.TorchTrainingPlan:
            if self.target is None and isinstance(self.dataset, Dataset):
                # Create Dataset for pytorch
                return TorchDataManager(dataset=self.dataset, **self.loader_arguments)
            elif isinstance(self.dataset, (pd.DataFrame, pd.Series, np.ndarray)) and \
                    isinstance(self.target, (pd.DataFrame, pd.Series, np.ndarray)):
                # If `dataset` and `target` attributes are array-like object
                # create TorchTabularDataset object to instantiate a TorchDataManager
                torch_dataset = TorchTabularDataset(inputs=self.dataset, target=self.target)
                return TorchDataManager(dataset=torch_dataset, **self.loader_arguments)
            else:
                raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Invalid arguments for torch based "
                                                f"training plan, either provide the argument  `dataset` as PyTorch "
                                                f"Dataset instance, or provide `dataset` and `target` arguments as "
                                                f"an instance one of pd.DataFrame, pd.Series or np.ndarray ")

        elif tp_type == TrainingPlans.SkLearnTrainingPlan:
            # Try to convert `torch.utils.Data.Dataset` to SkLearnBased dataset/datamanager
            if self.target is None and isinstance(self.dataset, Dataset):
                torch_data_manager = TorchDataManager(dataset=self.dataset)
                try:
                    sklearn_data_manager = torch_data_manager.to_sklearn()
                except Exception as e:
                    raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: PyTorch based `Dataset` object "
                                                    f"has been instantiated with DataManager. An error occurred while"
                                                    f"trying to convert torch.utils.data.Dataset to numpy based "
                                                    f"dataset: {str(e)}")
                return sklearn_data_manager

            # For scikit-learn based training plans, the arguments `dataset` and `target` should be an instance
            # one of `pd.DataFrame`, `pd.Series`, `np.ndarray`
            elif isinstance(self.dataset, (pd.DataFrame, pd.Series, np.ndarray)) and \
                    isinstance(self.target, (pd.DataFrame, pd.Series, np.ndarray)):
                # Create Dataset for SkLearn training plans
                return SkLearnDataManager(inputs=self.dataset, target=self.target, **self.loader_arguments)
            else:
                raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: The argument `dataset` and `target` "
                                                f"should be instance of pd.DataFrame, pd.Series or np.ndarray ")
        else:
            raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Undefined training plan")

        # elif self.target is not None and isinstance(self.dataset, Dataset):
        # raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: PyTorch based `Dataset` object "
        #                                 f"has been instantiated with DataManager while the target is not None. "
        #                                 f"This does not make sense for SkLearn based training plans. Either "
        #                                 f"provide `dataset` and `target` as an instance one of `pd.DataFrame` "
        #                                 f"`pd.Series` or `np.ndarray`, or just `dataset` as and instance of "
        #                                 f"`torch.utils.data.Dataset`.")


    # DISCUSS
    # def __getattr__(self, item):
    #
    #     """
    #     Wrap all functions/attributes of factory class members.
    #
    #     Args:
    #          item: Requested item from class
    #
    #     Raises:
    #         FedbiomedDataManagerError: If the attribute is not implemented
    #
    #     """
    #     try:
    #         return self._dataset_instance.__getattribute__(item)
    #     except AttributeError:
    #         raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: method {str(item)} not"
    #                                         f"implemented for class: {str(self._dataset_instance)}")
