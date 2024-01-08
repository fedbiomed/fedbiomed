# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
import inspect, os
import uuid
from abc import ABC
from re import findall
from typing import Any, Dict, List, Type, TypeVar, Union, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedJobError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
from fedbiomed.common.utils import import_class_from_file

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows._federated_workflow import exp_exceptions, FederatedWorkflow
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob, TrainingPlanApprovalJob
from fedbiomed.researcher.filetools import create_unique_link, choose_bkpt_file
from fedbiomed.researcher.secagg import SecureAggregation

# for checking class passed to experiment
training_plans_types = (TorchTrainingPlan, SKLearnTrainingPlan)
# typing information
TrainingPlan = TypeVar('TrainingPlan', TorchTrainingPlan, SKLearnTrainingPlan)
Type_TrainingPlan = TypeVar('Type_TrainingPlan', Type[TorchTrainingPlan], Type[SKLearnTrainingPlan])
TTrainingPlanWorkflow = TypeVar("TTrainingPlanWorkflow", bound='TrainingPlanWorkflow')  # only for typing


class TrainingPlanWorkflow(FederatedWorkflow, ABC):
    """
    A `TrainingPlanWorkflow` is an abstract entry point to orchestrate an experiment which uses a training plan.

    In addition to the functionalities provided by
    [`FederatedWorkflow`][fedbiomed.researcher.federated_workflows.FederatedWorkflow], the `TrainingPlanWorkflow` also
    manages the life-cycle of the training plan.

    !!! warning "Use `set_training_plan_class` to manage the training plan"
        Please only ever use the [`set_training_plan_class`][fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingPlanWorkflow.set_training_plan_class]
        function to manage the training plan. Do not set the training plan or training plan class directly!

    """

    @exp_exceptions
    def __init__(
            self,
            tags: Optional[Union[List[str], str]] = None,
            nodes: Optional[List[str]] = None,
            training_data: Optional[Union[FederatedDataSet, dict]] = None,
            training_plan_class: Optional[Type_TrainingPlan] = None,
            training_args: Optional[Union[TrainingArgs, dict]] = None,
            model_args: Optional[Dict] = None,
            experimentation_folder: Optional[str] = None,
            secagg: Union[bool, SecureAggregation] = False,
            save_breakpoints: bool = False,
    ) -> None:
        """Constructor of the class.

        Args:
            tags: list of string with data tags or string with one data tag. Empty list of tags ([]) means any dataset
                is accepted, it is different from None (tags not set, cannot search for training_data yet).
            nodes: list of node_ids to filter the nodes to be involved in the experiment. Defaults to None (no
                filtering).
            training_data:
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object from the dict and use this value as
                    training_data. The dict should use node ids as keys, values being list of dicts (each dict
                    representing a dataset on a node).
                * else if it is None (no training data provided)
                  - if `tags` is not None, set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                  - if `tags` is None, set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)
                Defaults to None (query nodes for dataset if `tags` is not None, set training_data
                to None else)
            training_plan_class: training plan class to be used for training.
                For experiment to be properly and fully defined `training_plan_class` needs to be a `Type_TrainingPlan`
                Defaults to None (no training plan class defined yet)
            model_args: contains model arguments passed to the constructor of the training plan when instantiating it :
                output and input feature dimension, etc.
            training_args: contains training arguments passed to the `training_routine` of the training plan when
                launching it: lr, epochs, batch_size...
            save_breakpoints: whether to save breakpoints or not after each training round. Breakpoints can be used for
                resuming a crashed experiment.
            experimentation_folder: choose a specific name for the folder where experimentation result files and
                breakpoints are stored. This should just contain the name for the folder not a path. The name is used
                as a subdirectory of `environ[EXPERIMENTS_DIR])`. Defaults to None (auto-choose a folder name)
                - Caveat : if using a specific name this experimentation will not be automatically detected as the last
                experimentation by `load_breakpoint`
                - Caveat : do not use a `experimentation_folder` name finishing with numbers ([0-9]+) as this would
                confuse the last experimentation detection heuristic by `load_breakpoint`.
            secagg: whether to setup a secure aggregation context for this experiment, and use it
                to send encrypted updates from nodes to researcher. Defaults to `False`
        """
        super().__init__(
            tags=tags,
            nodes=nodes,
            training_data=training_data,
            training_args=training_args,
            experimentation_folder=experimentation_folder,
            secagg=secagg,
            save_breakpoints=save_breakpoints
        )

        # Check arguments
        if training_plan_class is not None and not inspect.isclass(training_plan_class):
            msg = f"{ErrorNumbers.FB418.value}: bad type for argument `training_plan_class` {type(training_plan_class)}"
            raise FedbiomedJobError(msg)

        if training_plan_class is not None and not issubclass(training_plan_class, training_plans_types):
            msg = f"{ErrorNumbers.FB418.value}: bad type for argument `training_plan_class`. It is not subclass of " + \
                  f" supported training plans {training_plans_types}"
            raise FedbiomedJobError(msg)

        # __training_plan_class is the source of truth for all training plan members of this class
        # if __training_plan_class is None, then all other members are undefined
        # whenever __training_plan_class is changed, all other members should be immediately updated accordingly
        self.__training_plan_class = None
        # The __training_plan attribute represents the *actual instance* of a __training_plan_class that is currently
        # being used in the workflow
        self.__training_plan = None
        self._model_args = None
        self._global_model = None

        self.set_model_args(model_args)
        self.set_training_plan_class(training_plan_class)

    @exp_exceptions
    def _reset_training_plan(self) -> None:
        """Private utility function that resets the training plan according to the value of training plan class.

        If training plan class is None, then sets the training plan to None.
        Otherwise, it sets the training plan to a new, default-constructed instance of training plan class.
        """
        if self.__training_plan_class is None:
            self.__training_plan = None
            self._global_model = None
        else:
            self._raise_for_missing_job_prerequities()
            job = TrainingJob(reqs=self._reqs,
                              keep_files_dir=self.experimentation_path())
            self.__training_plan = job.get_initialized_tp_instance(self.training_plan_class(),
                                                                   self._training_args,
                                                                   self._model_args)
            self._global_model = self.__training_plan.after_training_params()

    def _raise_for_missing_job_prerequities(self) -> None:
        """Setter for job, it verifies pre-requisites are met for creating a job
        attached to this experiment. If yes, instantiate a job ; if no, return None.

        """
        super()._raise_for_missing_job_prerequities()
        # Check arguments
        if self.__training_plan_class is not None and not inspect.isclass(self.__training_plan_class):
            msg = f"{ErrorNumbers.FB418.value}: bad type for argument `training_plan_class` " \
                  f"{type(self.__training_plan_class)}"
            raise FedbiomedJobError(msg)

        if self.__training_plan_class is not None and not issubclass(self.__training_plan_class, training_plans_types):
            msg = f"{ErrorNumbers.FB418.value}: bad type for argument `training_plan_class`. It is not subclass of " + \
                  f" supported training plans {training_plans_types}"
            raise FedbiomedJobError(msg)

    @exp_exceptions
    def training_plan_class(self) -> Optional[Type_TrainingPlan]:
        """Retrieves the type of the training plan that is created for training.

        Please see also [`set_training_plan_class`][fedbiomed.researcher.experiment.Experiment.set_training_plan_class].

        Returns:
            training_plan_class: the class type of the training plan.
        """

        return self.__training_plan_class

    @exp_exceptions
    def training_plan(self) -> Optional[TrainingPlan]:
        """Retrieves the training plan instance currently being used in the federated workflow.

        Returns:
            training plan: an instance of one of [training_plans][fedbiomed.common.training_plans].
        """
        return self.__training_plan

    @exp_exceptions
    def model_args(self) -> dict:
        """Retrieves model arguments.

        Please see also [`set_model_args`][fedbiomed.researcher.experiment.Experiment.set_model_args]

        Returns:
            The arguments that are going to be passed to [`training_plans`][fedbiomed.common.training_plans]
                classes in built time on the node side.
        """
        return self._model_args

    @exp_exceptions
    def info(self, info=None) -> Dict[str, Any]:
        """Prints out the information about the current status of the experiment.

        Lists  all the parameters/arguments of the experiment and informs whether the experiment can be run.

        Raises:
            FedbiomedExperimentError: Inconsistent experiment due to missing variables
        """
        # at this point all attributes are initialized (in constructor)
        if info is None:
            info = {
                'Arguments': [],
                'Values': []
            }
        info['Arguments'].extend([
                'Training Plan Class',
                'Model Arguments',
            ])
        info['Values'].extend(['\n'.join(findall('.{1,60}',
                                         str(e))) for e in [
                           self.__training_plan_class,
                           self._model_args,
                       ]])
        info = super().info(info)
        return info

    @exp_exceptions
    def set_training_plan_class(self, training_plan_class: Union[Type_TrainingPlan, None]) -> \
            Union[Type_TrainingPlan, None]:
        """Sets  the training plan type + verification on arguments type

        !!! warning "Resets the training plan"
            This function has an important (and intended!) side-effect: it resets the `training_plan` attribute
            to a default-constructed instance.

        Args:
            training_plan_class: training plan class to be used for training.
                For experiment to be properly and fully defined `training_plan_class` needs to be a `Type_TrainingPlan`
                Defaults to None (no training plan class defined yet)

        Returns:
            `training_plan_class` that is set for experiment

        Raises:
            FedbiomedExperimentError : bad training_plan_class type
        """
        if training_plan_class is None:
            self.__training_plan_class = None
        elif inspect.isclass(training_plan_class):
            # training_plan_class must be a subclass of a valid training plan
            if issubclass(training_plan_class, training_plans_types):
                # valid class
                self.__training_plan_class = training_plan_class
            else:
                # bad class
                msg = ErrorNumbers.FB410.value + f' `training_plan_class` : {training_plan_class} class'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + f' `training_plan_class` of type: {type(training_plan_class)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        self._reset_training_plan()  # resets the training plan attribute

        return self.__training_plan_class

    @exp_exceptions
    def set_model_args(self, model_args: dict) -> dict:
        """Sets `model_args` + verification on arguments type

        Args:
            model_args (dict): contains model arguments passed to the constructor
                of the training plan when instantiating it : output and input feature
                dimension, etc.

        Returns:
            Model arguments that have been set.

        Raises:
            FedbiomedExperimentError : bad model_args type
        """
        if model_args is None or isinstance(model_args, dict):
            self._model_args = model_args
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + f' `model_args` : {type(model_args)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._model_args always exist at this point

        return self._model_args

    @exp_exceptions
    def check_training_plan_status(self) -> Dict:
        """ Method for checking training plan status, ie whether it is approved or not by the nodes

        Returns:
            Training plan status for answering nodes
        """
        job = TrainingPlanApprovalJob(reqs=self._reqs,
                                      nodes=self.training_data().node_ids(),
                                      keep_files_dir=self.experimentation_path())
        responses = job.check_training_plan_is_approved_by_nodes(job_id=self._id,
                                                                 training_plan=self.training_plan()
                                                                 )
        return responses

    @exp_exceptions
    def training_plan_approve(self,
                              description: str = "no description provided") -> dict:
        """Send a training plan and a ApprovalRequest message to node(s).

        This is a simple redirect to the Requests.training_plan_approve() method.

        If a list of node id(s) is provided, the message will be individually sent
        to all nodes of the list.
        If the node id(s) list is None (default), the message is broadcast to all nodes.

        Args:
            training_plan: the training plan to upload and send to the nodes for approval.
                   It can be:
                   - a path_name (str)
                   - a training_plan (class)
                   - an instance of a training plan
            nodes: list of nodes (specified by their UUID)
            description: Description for training plan approve request
            timeout: maximum waiting time for the answers

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the training plan has been correctly downloaded on the node side.
            Warning: status does not mean that the training plan is approved, only that it has been added
            to the "approval queue" on the node side.
        """
        job = TrainingPlanApprovalJob(reqs=self._reqs,
                                      nodes=self.training_data().node_ids(),
                                      keep_files_dir=self.experimentation_path())
        responses = job.training_plan_approve(training_plan=self.training_plan(),
                                              description=description,
                                              )
        return responses
    @exp_exceptions
    def breakpoint(self,
                   state,
                   bkpt_number) -> None:
        """
        Saves breakpoint with the state of the training at a current round. The following Experiment attributes will
        be saved:
          - training_plan_class
          - model_args
        """
        # save training plan to file
        training_plan_module = 'model_' + str(uuid.uuid4())
        training_plan_file = os.path.join(self.experimentation_path(), training_plan_module + '.py')
        self.training_plan().save_code(training_plan_file)

        state.update({
            'model_args': self._model_args,
            'training_plan_path': training_plan_file,
            'training_plan_class_name': self.__training_plan_class.__name__,
        })

        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(self._experimentation_folder, bkpt_number - 1)

        # rewrite paths in breakpoint : use the links in breakpoint directory
        state['training_plan_path'] = create_unique_link(
            breakpoint_path,
            # - Need a file with a restricted characters set in name to be able to import as module
            'model_' + str("{:04d}".format(bkpt_number - 1)), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('..', os.path.basename(state["training_plan_path"]))
        )

        super().breakpoint(state, bkpt_number)


    @classmethod
    @exp_exceptions
    def load_breakpoint(cls,
                        breakpoint_folder_path: Union[str, None] = None) -> 'TTrainingPlanWorkflow':
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
          cls: Experiment class
          breakpoint_folder_path: path of the breakpoint folder. Path can be absolute or relative eg:
            "var/experiments/Experiment_xxxx/breakpoints_xxxx". If None, loads latest breakpoint of the latest
            experiment. Defaults to None.

        Returns:
            Reinitialized experiment object. With given object-0.2119,  0.0796, -0.0759, user can then use `.run()`
                method to pursue model training.

        Raises:
            FedbiomedExperimentError: bad argument type, error when reading breakpoint or bad loaded breakpoint
                content (corrupted)
        """
        loaded_exp, saved_state = super().load_breakpoint()

        # Import TP class
        _, tp_class = import_class_from_file(
            module_path=saved_state.get("training_plan_path"),
            class_name=saved_state.get("training_plan_class_name")
        )

        loaded_exp.set_model_args(saved_state["model_args"])
        loaded_exp.set_training_plan_class(tp_class)
        training_plan = loaded_exp.training_plan()
        if training_plan is None:
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                  'breakpoint file seems corrupted, `training_plan` is None'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return loaded_exp, saved_state
