""" This file contains dummy Classes for unit testing. It fakes a TrainingPlan
(either from `fedbiomed.common.torchnn`` or from `fedbiomed.common.fedbiosklearn`)
"""

from typing import Dict, Any, List
import time


# Fakes TrainingPlan (either `fedbiomed.common.torchnn`` or `fedbiomed.common.fedbiosklearn`)
class FakeModel:
    """Fakes a model (TrainingPlan, inheriting either from 
    `fedbiomed.common.torchnn` or from `fedbiomed.common.fedbiosklearn`)
    Provides a few methods that mimicks the behaviour of 
    TrainingPlan models

    """
    SLEEPING_TIME = 1  # time that simulate training (in seconds)

    def __init__(self, model_args: Dict = None, *args, **kwargs):
        # For testing Job model_args
        self.model_args = model_args
        self.__type = 'DummyTrainingPlan'
        self._optimizer_args = {}
        pass

    def post_init(self, model_args, training_args, optimizer_args=None, aggregator_args=None):
        pass

    def type(self):
        """ Getter for TrainingPlan Type"""
        return self.__type

    def set_data_loaders(self, train_data_loader, test_data_loader):
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

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
        pass

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
        pass

    def save_code(self, path: str):
        """
        Fakes `save_code` method of TrainingPlan classes, originally used for
        saving codes of model calss. Passed argument are unused.

        Args:
            path (str): saving path
        """
        pass

    def set_dataset_path(self, path: str):
        """Fakes `set_dataset` method of TrainingPlan classes. Originally 
        used for setting dataset path. Passed arguments are unused.        

        Args:
            path (str): originally, path where the node dataset are stored.
            Unused in this method.
        """
        pass

    def optimizer_args(self):
        return self._optimizer_args

    def training_routine(self, **kwargs):
        """Fakes `training_routine` method of TrainingPlan classes. Originally
        used for training the model. Passed arguments are unused.
        Sleeps for a certain amount of time (set by SLEEPING_TIME attibute),
        so it mimicks a training and able timing tests.
        """
        time.sleep(FakeModel.SLEEPING_TIME)

    def testing_routine(self, metric, history_monitor, before_train: bool):
        pass

    def after_training_params(self) -> List[int]:
        """Fakes `after_training_params` method of TrainingPlan classes.
        Originally used to get the parameters after training is performed.
        Passed arguments are unused.
        
        Returns:
            List[int]: Mimicks return of trained parameters 
            (always returns a list of integers: [1, 2, 3, 4])
        """
        return [1, 2, 3, 4]
