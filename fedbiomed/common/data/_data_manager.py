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

        Raises:

            - FedbiomedDataManagerError: - If `target` is not None and `dataset` is an instance of
                                          `torch.utils.data.Dataset`. This scenario does not make sense, since
                                          `Dataset` object has already target variable in it.

                                         - If `target` is not None and `dataset` or `target` is not an instance one
                                            of `pd.DataFrame`, `pd.Series` or `np.ndarray`.
        """

        # TODO: Improve datamanager for auto loading by given dataset_path and other information
        # such as inputs variable indexes and target variables indexes

        self.dataset = dataset
        self.target = target
        self.loader_arguments = kwargs

    def load(self, tp_type: TrainingPlans):

        # TorchDataset object shouldbe instantiated if target variable is not defined
        # and `dataset` is an instance of `torch.utils.data.Dataset`

        if tp_type == TrainingPlans.TorchTrainingPlan:

            if self.target is None and isinstance(self.dataset, Dataset):
                # Create Dataset for pytorch
                return TorchDataManager(dataset=self.dataset, **self.loader_arguments)
            elif isinstance(self.dataset, (pd.DataFrame, pd.Series, np.ndarray)) and \
                    isinstance(self.target, (pd.DataFrame, pd.Series, np.ndarray)):
                torch_dataset = TorchTabularDataset(inputs=self.dataset, target=self.target)
                return TorchDataManager(dataset=torch_dataset, **self.loader_arguments)
            else:
                raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: Invalid arguments for torch based "
                                                f"training plan, either provide the argument  `dataset` as PyTorch "
                                                f"Dataset instance, or provide `dataset` and `target` arguments as "
                                                f"an instance one of pd.DataFrame, pd.Series or np.ndarray ")

        elif tp_type == TrainingPlans.SkLearnTrainingPlan:

            # Fed-BioMed framework uses PyTorch Dataset object to train PyTorch based models and researcher
            # is responsible for providing Dataset object in training plan. Since `torch.utils.data.Dataset`
            # is always instantiated with target variables, passing the argument `target` is as not None does
            # not make sense. The argument `target` only used for scikit-learn training. Therefore, if target is not
            # None `inputs` should be an instance of np.ndarray pd.DataFrame or pd.Series.
            # It means that the arguments `dataset` (independent variables) and `target` (dependent variable)
            # will be used for SkLearn models.

            if isinstance(self.dataset, Dataset):
                raise FedbiomedDataManagerError(f"{ErrorNumbers.FB607.value}: The `target` argument has been "
                                                f"passed while the argument `dataset` is an instance of "
                                                f"PyTorch Dataset. Either instantiate your `target` variable in"
                                                f"your `Dataset` object to used in `TorchTrainingPlan` or pass "
                                                f"dataset as an instance of `pd.Dataframe` , `pd.Series` or "
                                                f"`np.ndarray` to use scikit-learn based training plan. ")

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
