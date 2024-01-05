# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
import inspect, os
import uuid
from abc import ABC
from re import findall
from pathvalidate import sanitize_filepath
from typing import Any, Dict, List, Type, TypeVar, Union, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedJobError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
from fedbiomed.common.utils import import_class_from_file

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows._federated_workflow import exp_exceptions, FederatedWorkflow
from fedbiomed.researcher.federated_workflows.jobs import Job, TrainingPlanApprovalJob
from fedbiomed.researcher.filetools import create_unique_link, choose_bkpt_file
from fedbiomed.researcher.secagg import SecureAggregation

# for checking class passed to experiment
# TODO : should we move this to common/constants.py ?
training_plans_types = (TorchTrainingPlan, SKLearnTrainingPlan)
# for typing only
TrainingPlan = TypeVar('TrainingPlan', TorchTrainingPlan, SKLearnTrainingPlan)
Type_TrainingPlan = TypeVar('Type_TrainingPlan', Type[TorchTrainingPlan], Type[SKLearnTrainingPlan])


class TrainingPlanWorkflow(FederatedWorkflow, ABC):
    """
    This class represents the orchestrator managing the federated training
    """

    @exp_exceptions
    def __init__(
            self,
            tags: Union[List[str], str, None] = None,
            nodes: Union[List[str], None] = None,
            training_data: Union[FederatedDataSet, dict, None] = None,
            training_plan_class: Union[Type_TrainingPlan, str, None] = None,
            training_plan_path: Union[str, None] = None,
            training_args: Union[TrainingArgs, dict, None] = None,
            experimentation_folder: Union[str, None] = None,
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
            aggregator: object or class defining the method for aggregating local updates. Default to None (use
                [`FedAverage`][fedbiomed.researcher.aggregators.FedAverage] for aggregation)
            agg_optimizer: [`Optimizer`][fedbiomed.common.optimizers.Optimizer] instance, to refine aggregated
                model updates prior to their application. If None, merely apply the aggregated updates.
            node_selection_strategy:object or class defining how nodes are sampled at each round for training, and how
                non-responding nodes are managed.  Defaults to None:
                - use [`DefaultStrategy`][fedbiomed.researcher.strategies.DefaultStrategy] if training_data is
                    initialized
                - else strategy is None (cannot be initialized), experiment cannot be launched yet
            round_limit: the maximum number of training rounds (nodes <-> central server) that should be executed for
                the experiment. `None` means that no limit is defined. Defaults to None.
            training_plan_class: name of the training plan class [`str`][str] or training plan class
                (`Type_TrainingPlan`) to use for training.
                For experiment to be properly and fully defined `training_plan_class` needs to be:
                - a [`str`][str] when `training_plan_class_path` is not None (training plan class comes from a file).
                - a `Type_TrainingPlan` when `training_plan_class_path` is None (training plan class passed
                    as argument).
                Defaults to None (no training plan class defined yet)

            training_plan_path: path to a file containing training plan code [`str`][str] or None (no file containing
                training plan code, `training_plan` needs to be a class matching `Type_TrainingPlan`) Defaults to None.
            model_args: contains model arguments passed to the constructor of the training plan when instantiating it :
                output and input feature dimension, etc.
            training_args: contains training arguments passed to the `training_routine` of the training plan when
                launching it: lr, epochs, batch_size...
            save_breakpoints: whether to save breakpoints or not after each training round. Breakpoints can be used for
                resuming a crashed experiment.
            tensorboard: whether to save scalar values  for displaying in Tensorboard during training for each node.
                Currently, it is only used for loss values.
                - If it is true, monitor instantiates a `Monitor` object that write scalar logs into `./runs` directory.
                - If it is False, it stops monitoring if it was active.
            experimentation_folder: choose a specific name for the folder where experimentation result files and
                breakpoints are stored. This should just contain the name for the folder not a path. The name is used
                as a subdirectory of `environ[EXPERIMENTS_DIR])`. Defaults to None (auto-choose a folder name)
                - Caveat : if using a specific name this experimentation will not be automatically detected as the last
                experimentation by `load_breakpoint`
                - Caveat : do not use a `experimentation_folder` name finishing with numbers ([0-9]+) as this would
                confuse the last experimentation detection heuristic by `load_breakpoint`.
            secagg: whether to setup a secure aggregation context for this experiment, and use it
                to send encrypted updates from nodes to researcher. Defaults to `False`

        Raises:
            FedbiomedJobError: bad argument type or value
            FedbiomedJobError: cannot save training plan to file

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

        # predefine all class variables, so no need to write try/except
        # block each time we use it
        self._training_plan_path = None
        self._training_plan = None
        self._training_plan_file: Optional[str] = None
        # __training_plan_class is the source of truth for all training plan members of this class
        # if __training_plan_class is None, then all other members are undefined
        # whenever __training_plan_class is changed, all other members should be immediately updated accordingly
        self.__training_plan_class = None

        self.set_training_plan_class(training_plan_class)
        self.set_training_plan_path(training_plan_path)
        self.reset_training_plan()

    @exp_exceptions
    def reset_training_plan(self):
        if self.__training_plan_class is None:
            self._training_plan = None
        else:
            self._raise_for_missing_job_prerequities()
            job = Job(reqs=self._reqs,
                      keep_files_dir=self.experimentation_path())
            self._training_plan = job.get_default_constructed_tp_instance(self.__training_plan_class)

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
    def training_plan_class(self) -> Union[Type_TrainingPlan, str, None]:
        """Retrieves the training plan (training plan class) that is created for training.

        Please see also [`set_training_plan_class`][fedbiomed.researcher.experiment.Experiment.set_training_plan_class].

        Returns:
            Training plan class as one of [`Type_TrainingPlan`][fedbiomed.researcher.experiment.Type_TrainingPlan]. None
                if it isn't declared yet. [`str`][str] if [`training_plan_path`]
                [fedbiomed.researcher.experiment.Experiment.training_plan_path]that represents training plan class
                created externally is provided.
        """

        return self.__training_plan_class

    @exp_exceptions
    def training_plan_path(self) -> Union[str, None]:
        """Retrieves training plan path where training plan class is saved as python script externally.

        Please see also [`set_training_plan_path`][fedbiomed.researcher.experiment.Experiment.set_training_plan_path].

        Returns:
            Path to python script (`.py`) where training plan class (training plan) is created. None if it isn't
                declared yet.
        """

        return self._training_plan_path

    @exp_exceptions
    def training_plan(self) -> Union[TrainingPlan, None]:
        """ Retrieves training plan instance that has been built and send the nodes through HTTP restfull service
        for each round of training.

        !!! info "Loading aggregated parameters"
            After retrieving the training plan instance aggregated parameters should be loaded.
            Example:
            ```python
            training_plan = exp.training_plan()
            training_plan.model.load_state_dict(exp.aggregated_params()[rounds - 1]['params'])
            ```

        Returns plan object which is an instance one of [training_plans][fedbiomed.common.training_plans].
        """
        return self._training_plan

    # a specific getter-like
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
                'Training Plan Path',
                'Training Plan Class',
            ])
        info['Values'].extend(['\n'.join(findall('.{1,60}',
                                         str(e))) for e in [
                           self._training_plan_path,
                           self.__training_plan_class,
                       ]])
        info = super().info(info)
        return info

    @exp_exceptions
    def set_training_plan_class(self, training_plan_class: Union[Type_TrainingPlan, str, None]) -> \
            Union[Type_TrainingPlan, str, None]:
        """Sets  `training_plan` + verification on arguments type

        Args:
            training_plan_class: name of the training plan class (`str`) or training plan class as one
                of [`TrainingPlans`] [fedbiomed.common.training_plans] to use for training.
                For experiment to be properly and fully defined `training_plan_class` needs to be:
                    - a `str` when `training_plan_path` is not None (training plan class comes from a file).
                    - a `Type_TrainingPlan` when `training_plan_path` is None (training plan class passed
                    as argument).

        Returns:
            `training_plan_class` that is set for experiment

        Raises:
            FedbiomedExperimentError : bad training_plan_class type
        """
        if training_plan_class is None:
            self.__training_plan_class = None
            self._training_plan_is_defined = False
        elif isinstance(training_plan_class, str):
            if str.isidentifier(training_plan_class):
                # correct python identifier
                self.__training_plan_class = training_plan_class
                # training_plan_class_path may not be defined at this point

                self._training_plan_is_defined = isinstance(self._training_plan_path, str)

            else:
                # bad identifier
                msg = ErrorNumbers.FB410.value + f' `training_plan_class` : {training_plan_class} bad identifier'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        elif inspect.isclass(training_plan_class):
            # training_plan_class must be a subclass of a valid training plan
            if issubclass(training_plan_class, training_plans_types):
                # valid class
                self.__training_plan_class = training_plan_class
                # training_plan_class_path may not be defined at this point

                self._training_plan_is_defined = self._training_plan_path is None
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

            # self._training_plan_is_defined and self.__training_plan_class always exist at this point
        if not self._training_plan_is_defined:
            logger.debug(f'Experiment not fully configured yet: no valid training plan, '
                         f'training_plan_class={self.__training_plan_class} '
                         f'training_plan_class_path={self._training_plan_path}')

        self.reset_training_plan()

        return self.__training_plan_class

    @exp_exceptions
    def set_training_plan_path(self, training_plan_path: Union[str, None]) -> Union[str, None]:
        """Sets `training_plan_path` + verification on arguments type.

        Training plan path is the path where training plan class is saved as python script/module externally.

        Args:
            training_plan_path (Union[str, None]) : path to a file containing  training plan code (`str`) or None
                (no file containing training plan code, `training_plan` needs to be a class matching one
                of [`training_plans`][fedbiomed.common.training_plans]

        Returns:
            The path that is set for retrieving module where training plan class is defined

        Raises:
            FedbiomedExperimentError : bad training_plan_path type
        """
        # self._training_plan and self._training_plan_is_defined already exist when entering this function

        if training_plan_path is None:
            self._training_plan_path = None
            # .. so training plan is defined if it is a class (+ then, it has been tested as valid)
            self._training_plan_is_defined = inspect.isclass(self.__training_plan_class)
        elif isinstance(training_plan_path, str):
            if sanitize_filepath(training_plan_path, platform='auto') == training_plan_path \
                    and os.path.isfile(training_plan_path):
                # provided training plan path is a sane path to an existing file
                self._training_plan_path = training_plan_path
                # if providing a training plan path, we expect a training plan class name (not a class)
                self._training_plan_is_defined = isinstance(self.__training_plan_class, str)
            else:
                # bad filepath
                msg = ErrorNumbers.FB410.value + \
                      f' `training_plan_path` : {training_plan_path} is not a path to an existing file'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + ' `training_plan_path` must be string, ' \
                                             f'but got type: {type(training_plan_path)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # self._training_plan_path is also defined at this point
        if not self._training_plan_is_defined:
            logger.debug(f'Experiment not fully configured yet: no valid training plan, '
                         f'training_plan={self.__training_plan_class} training_plan_path={self._training_plan_path}')

        return self._training_plan_path

    # Training plan checking functions
    @exp_exceptions
    def training_plan_file(self, display: bool = True) -> str:
        """ This method displays saved final training plan for the experiment
            that will be sent to the nodes for training.

        Args:
            display: If `True`, prints content of the training plan file. Default is `True`

        Returns:
            Path to training plan file

        Raises:
            FedbiomedExperimentError: bad argument type, or cannot read training plan file content
        """
        if not isinstance(display, bool):
            # bad type
            msg = ErrorNumbers.FB410.value + \
                  f', in method `training_plan_file` param `display` : type {type(display)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # Display content so researcher can copy
        try:
            if display:
                with open(self._training_plan_file) as file:
                    content = file.read()
                    file.close()
                    print(content)
        except OSError as e:
            # cannot read training plan file content
            msg = ErrorNumbers.FB412.value + \
                  f', in method `training_plan_file` : error when reading training plan file - {e}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._training_plan_file

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
          - round_current
          - round_limit
          - tags
          - experimentation_folder
          - aggregator
          - agg_optimizer
          - node_selection_strategy
          - training_data
          - training_args
          - model_args
          - training_plan_path
          - training_plan_class
          - aggregated_params
          - job (attributes returned by the Job, aka job state)
          - secagg

        Raises:
            FedbiomedExperimentError: experiment not fully defined, experiment did not run any round yet, or error when
                saving breakpoint
        """
        # save training plan to file
        training_plan_module = 'model_' + str(uuid.uuid4())
        self._training_plan_file = os.path.join(self.experimentation_path(), training_plan_module + '.py')
        self.training_plan().save_code(self._training_plan_file)

        state.update({
            'training_plan_path': self._training_plan_file,
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
                        breakpoint_folder_path: Union[str, None] = None) -> 'TExperiment':
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Usefull if training has crashed
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

        loaded_exp.set_training_plan_class(tp_class)
        training_plan = loaded_exp.training_plan()
        if training_plan is None:
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                  'breakpoint file seems corrupted, `training_plan` is None'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return loaded_exp, saved_state
