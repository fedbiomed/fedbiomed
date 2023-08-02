"""Fake `BaseTrainingPlan` subclass nullifying most methods, for tests use."""

from typing import Any, Dict, Optional
from unittest import mock
import time
from fedbiomed.common.constants import TrainingPlans
import torch
from torch.utils.data import Dataset, DataLoader
from fedbiomed.common.models import Model
from fedbiomed.common.optimizers import BaseOptimizer
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.data import DataManager

# Fakes TrainingPlan (either `fedbiomed.common.torchnn`` or `fedbiomed.common.fedbiosklearn`)
class FakeModel(BaseTrainingPlan):
    """Fake `BaseTrainingPlan` subclass nullifying most methods.

    This class is designed to be used in the context of tests.
    Important notes:
      - Its wrapped `Model` is an auto-spec mock object.
      - The latter deterministically returns a list of int when queried
        for its model parameters, that have values [1, 2, 3, 4].
      - The `training_routine` method implements a 1-second sleep action.
    """
    SLEEPING_TIME = 1  # time that simulate training (in seconds)

    def __init__(self, model_args: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        # For testing Job model_args
        self.model_args = model_args
        self.__type = 'DummyTrainingPlan'
        self._optimizer_args = {}
        self._model = mock.create_autospec(Model, instance=True)
        self._model.get_weights.return_value = {"coefs": [1, 2, 3, 4]}
        self._optimizer = mock.create_autospec(BaseOptimizer, instance=True)
        self._optimizer.optimizer = mock.MagicMock()

    def post_init(
        self,
        model_args: Dict[str, Any],
        training_args: Dict[str, Any],
        aggregator_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fake 'post_init', that does not make use of input arguments."""
        return None

    def model(self):
        return self._model

    def type(self):
        """Getter for TrainingPlan Type."""
        return self.__type

    def load(self, path: str, to_params: bool):
        """Fakes `load` method of TrainingPlan classes,
        used for loading model parameters. API mimickes
        TrainingPlan 's `load` method, but passed arguments
        are unused

        Args:
            path (str): originally, the path where the parameters model
            are stored. Unused in this dummy class.
            to_params (bool): originally, whether to return parameter into
            the model or into a dictionary. Unused in this dummy class.
        """
    def init_optimizer(self, optimizer_args: Dict[str, Any]):
        """Fakes `init_optimizer` method, used to initialize an optimizer (either framework
        specific like pytorch optimizer, or non-framework specific like declearn)

        Args:
            optimizer_args: optimizer parameters as a dictionary
        """

    def optimizer(self):
        return self._optimizer

    def save(self, filename: str, results: Dict[str, Any] = None):
        """
        Fakes `save` method of TrainingPlan classes, originally used for
        saving node's local model. Passed argument are unused.

        Args:
            filename (str): originally, the name of the file
            that will contain the saved parameters. Unused in this
            dummy class.
            results (Dict[str, Any]): originally, contains the
            results of the training. Unused in this method.
        """

    def save_code(self, path: str):
        """
        Fakes `save_code` method of TrainingPlan classes, originally used for
        saving codes of model calss. Passed argument are unused.

        Args:
            path (str): saving path
        """

    def set_dataset_path(self, path: str):
        """Fakes `set_dataset` method of TrainingPlan classes. Originally
        used for setting dataset path. Passed arguments are unused.

        Args:
            path (str): originally, path where the node dataset are stored.
            Unused in this method.
        """

    def training_routine(self, **kwargs):
        """Fakes `training_routine` method of TrainingPlan classes. Originally
        used for training the model. Passed arguments are unused.
        Sleeps for a certain amount of time (set by SLEEPING_TIME attibute),
        so it mimicks a training and able timing tests.
        """
        time.sleep(FakeModel.SLEEPING_TIME)

    def testing_routine(self, metric, history_monitor, before_train: bool):
        pass


class DeclearnAuxVarModel(FakeModel):
    """for specific test that  tests declearn specific optimizer compatibility"""
    OPTIM = None
    TYPE = None

    class CustomDataset(Dataset):
        def __init__(self, *args):
            self.value = torch.Tensor([[1, 2, 3], [1, 2, 3]])
            self.target = torch.Tensor([1, 1])

        def __getitem__(self, index):
            return self.value[index], self.target[index]

        def __len__(self):
            return self.target.shape[0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # overriding specific component 
        self._optimizer = DeclearnAuxVarModel.OPTIM

    def training_data(self):
        return DataManager(dataset=self.CustomDataset())
    
    def type(self):
        return DeclearnAuxVarModel.TYPE
    
    def training_routine(self, **kwargs):
        td = self.training_data()
        td.load(TrainingPlans.TorchTrainingPlan)
        all_s = td.load_all_samples()
        
        for v, t in all_s:
            o = self._optimizer._model.model(v)
            loss = t - o # stupid loss function, created only for the sake of testing
            loss.backward()
            self._optimizer.step()
        return super().training_routine(**kwargs)
