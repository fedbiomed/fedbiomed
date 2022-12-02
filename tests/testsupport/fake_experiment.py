""" This file contains dummy Classes for unit testing. It fakes Experiment class
(from fedbiomed.researcher.experiment)
"""
from typing import Union, TypeVar, Type, List

from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.researcher.experiment import Experiment

# need those types defined
FederatedDataSet = TypeVar("FederatedDataSet")
Aggregator = TypeVar("Aggregator")
Strategy = TypeVar("Strategy")
Type_TrainingPlan = TypeVar("Type_TrainingPlan")
TrainingPlan = TypeVar("TrainingPlan")
_E = TypeVar("Experiment")

class ExperimentMock():
    """Provides an interface that behave like the Experiment,
    for a subset of methods
    """

    def __init__(self,
                tags: Union[List[str], str, None] = None,
                nodes: Union[List[str], None] = None,
                training_data: Union[FederatedDataSet, dict, None] = None,
                aggregator: Union[Aggregator, Type[Aggregator], None] = None,
                node_selection_strategy: Union[Strategy, Type[Strategy], None] = None,
                round_limit: Union[int, None] = None,
                training_plan_class: Union[Type_TrainingPlan, str, None] = None,
                training_plan_path: Union[str, None] = None,
                model_args: dict = {},
                training_args: dict = {},
                save_breakpoints: bool = False,
                tensorboard: bool = False,
                experimentation_folder: Union[str, None] = None
                ):
        """ Constructor of the class.

        Args and Returns : see original function
        """
        self._tags = tags
        self._nodes = nodes
        self._fds = training_data
        self._aggregator = aggregator
        self._node_selection_strategy = node_selection_strategy
        self._round_current = 0
        self._round_limit = round_limit
        self._experimentation_folder = experimentation_folder
        self._training_plan_class = training_plan_class
        self._training_plan_path = training_plan_path
        self._model_args = model_args
        self._training_args = TrainingArgs(only_required=False)
        self.aggregator_args = {}
        class Job:
            def load_state(self, saved_state):
                self._saved_state = saved_state

        self._job = Job() # minimal
        self._aggregated_params = {}
        self._save_breakpoints = save_breakpoints
        self._monitor = tensorboard # minimal


    def _set_round_current(self, round_current: int) -> int:
        """Set `round_current` in mocked class

        Args and Returns : see original function
        """
        self._round_current = round_current
        return self._round_current
