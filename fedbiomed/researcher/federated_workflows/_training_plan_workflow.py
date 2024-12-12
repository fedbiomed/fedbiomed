"""
This file is originally part of Fed-BioMed
SPDX-License-Identifier: Apache-2.0
"""
import os
import inspect
import uuid
from abc import ABC
from contextlib import contextmanager
from copy import deepcopy
from re import findall
from typing import Any, Dict, List, Type, TypeVar, Union, Optional, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedTypeError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan, TorchTrainingPlan, SKLearnTrainingPlan
from fedbiomed.common.utils import (
    import_class_from_file,
    import_class_object_from_file
)

from fedbiomed.researcher.federated_workflows.jobs \
    import TrainingPlanApproveJob, TrainingPlanCheckJob
from fedbiomed.researcher.filetools import create_unique_link, choose_bkpt_file

from ._federated_workflow import exp_exceptions, FederatedWorkflow

# for checking class passed to experiment
TRAINING_PLAN_TYPES = (TorchTrainingPlan, SKLearnTrainingPlan)
# typing information
TrainingPlan = TypeVar('TrainingPlan', TorchTrainingPlan, SKLearnTrainingPlan)
TrainingPlanT = TypeVar('TrainingPlanT', Type[TorchTrainingPlan], Type[SKLearnTrainingPlan])
TrainingPlanWorkflowT = \
    TypeVar("TrainingPlanWorkflowT", bound='TrainingPlanWorkflow')  # only for typing
T = TypeVar("T")


class TrainingPlanWorkflow(FederatedWorkflow, ABC):
    """ A `TrainingPlanWorkflow` is an abstract entry point to orchestrate
    an experiment which uses a training plan.

    In addition to the functionalities provided by
    [`FederatedWorkflow`][fedbiomed.researcher.federated_workflows.FederatedWorkflow],
    the `TrainingPlanWorkflow` also manages the life-cycle of the training plan.

    !!! warning "Use `set_training_plan_class` to manage the training plan"
        Please only ever use the
        [`set_training_plan_class`][fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingPlanWorkflow.set_training_plan_class]
        function to manage the training plan. Do not set the training plan
        or training plan class directly!
    """

    @exp_exceptions
    def __init__(
        self,
        *args,
        training_plan_class: Optional[TrainingPlanT] = None,
        training_args: Optional[Union[TrainingArgs, dict]] = None,
        model_args: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Constructor of the class.

        Args:
            training_plan_class: training plan class to be used for training.
                For experiment to be properly and fully defined `training_plan_class`
                needs to be a `TrainingPlanT` Defaults to None (no training plan class
                defined yet.
            model_args: contains model arguments passed to the constructor
                of the training plan when instantiating it :
                output and input feature dimension, etc.
            training_args: contains training arguments passed to the `training_routine`
                of the training plan when launching it: lr, epochs, batch_size...
            *args: Extra positional arguments from parent class
                [`FederatedWorkflow`][fedbiomed.researcher.federated_workflows.FederatedWorkflow]
            **kwargs: Arguments of parent class
                [`FederatedWorkflow`][fedbiomed.researcher.federated_workflows.FederatedWorkflow]
        """
        # Check arguments
        if training_plan_class is not None and not inspect.isclass(training_plan_class):
            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB410.value}: bad type for argument "
                f"`training_plan_class` {type(training_plan_class)}")

        if training_plan_class is not None and \
                not issubclass(training_plan_class, TRAINING_PLAN_TYPES):

            raise FedbiomedTypeError(
                f"{ErrorNumbers.FB410.value}: bad type for argument `training_plan_class`."
                f" It is not subclass of supported training plans {TRAINING_PLAN_TYPES}")

        # _training_plan_class determines the life-cycle of the training plan:
        # if training_plass_class changes, then the training plan must be reinitialized
        self._training_plan_class = None
        # model args is also tied to the life-cycle of training plan:
        # if model_args changes, the training plan must be reinitialized
        self._model_args = None
        # The _training_plan attribute represents the *actual instance*
        # of a _training_plan_class that is currently
        # being used in the workflow. The training plan cannot be modified by the user.
        self._training_plan = None
        self._training_args: Optional[TrainingArgs] = None  # FIXME: is it ok to have this here?
        # The _training_plan_file attribute represents the path of the file where the training plan is saved.
        # It cannot be modified by the user
        self._training_plan_file = None

        # initialize object
        super().__init__(*args, **kwargs)

        self.set_training_args(training_args)
        self.set_model_args(model_args)
        self.set_training_plan_class(training_plan_class)

    def _instantiate_training_plan(self) -> Tuple[BaseTrainingPlan, str]:
        """Instantiates training plan class

        Args:
            training_plan_class: Training plan class

        Returns:
            a tuple of an initialized training plan object, and the path of the file
                where the training plan is saved
        """

        # FIXME: Following actions can be part of training plan class
        # create TrainingPlan instance
        training_plan_class = self.training_plan_class()
        training_plan = training_plan_class()  # contains TrainingPlan

        # save and load training plan to a file to be sure
        # 1. a file is associated to training plan, so we can read its source, etc.
        # 2. all dependencies are applied
        training_plan_module = 'model_' + str(uuid.uuid4())
        training_plan_file = os.path.join(
            self.experimentation_path(),
            training_plan_module + '.py'
        )

        training_plan.save_code(training_plan_file)
        del training_plan

        _, training_plan = import_class_object_from_file(
            training_plan_file, training_plan_class.__name__)

        training_plan.post_init(
            model_args={} if self._model_args is None else self._model_args,
            training_args=self._training_args,
            initialize_optimizer=False
        )

        return training_plan, training_plan_file


    @exp_exceptions
    def _update_training_plan(self,
                              keep_weights: bool = True) -> None:
        """Private utility function that updates the training plan according to the value
        of training plan class.

        If training plan class is None, then sets the training plan to None.
        Otherwise, it sets the training plan to a new, default-constructed
        instance of training plan class.
        """

        if self._training_plan_class is None:
            self._training_plan = None
            self._training_plan_file = None
        else:
            with self._keep_weights(keep_weights):
                self._training_plan, self._training_plan_file = self._instantiate_training_plan()


    @exp_exceptions
    def training_plan_class(self) -> Optional[TrainingPlanT]:
        """Retrieves the type of the training plan that is created for training.

        Please see also
        [`set_training_plan_class`][fedbiomed.researcher.federated_workflows.TrainingPlanWorkflow.set_training_plan_class].

        Returns:
            training_plan_class: the class type of the training plan.
        """

        return self._training_plan_class


    @exp_exceptions
    def training_plan_file(self, display: bool = True) -> str:
        """Retrieves the path of the file where the training plan is saved, and optionally displays it.

        Args:
            display: If `True`, prints the content of the training plan file. Default is `True`

        Returns:
            Path to the training plan file

        Raises:
            FedbiomedExperimentError: bad argument type, or cannot read training plan file content
        """
        if not isinstance(display, bool):
            # bad type
            msg = ErrorNumbers.FB410.value + \
                f', in method `training_plan_file` param `display` : type {type(display)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        if display and self._training_plan_file is not None:
            try:
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
    def training_args(self) -> dict:
        """Retrieves training arguments.

        Please see also [`set_training_args`][fedbiomed.researcher.\
        federated_workflows.FederatedWorkflow.set_training_args]

        Returns:
            The arguments that are going to be passed to the training plan's
                `training_routine` to perfom training on the node side. An example
                training routine: [`TorchTrainingPlan.training_routine`]
                [fedbiomed.common.training_plans.TorchTrainingPlan.training_routine]
        """

        return self._training_args.dict()



    @exp_exceptions
    def training_plan(self) -> Optional[TrainingPlan]:
        """Retrieves the training plan instance currently being used in the federated workflow.

        Returns:
            training plan: the training plan instance
        """
        return self._training_plan

    @exp_exceptions
    def model_args(self) -> dict:
        """Retrieves model arguments.

        Please see also [`set_model_args`][fedbiomed.researcher.federated_workflows.TrainingPlanWorkflow.set_model_args]

        Returns:
            The arguments that are going to be passed to the `init_model` function of the training plan during
            initialization of the model instance
        """
        return self._model_args

    @exp_exceptions
    def info(
        self,
        info: Optional[Dict] = None,
        missing: str = ''
    ) -> Tuple[Dict[str, List[str]], str]:
        """Prints out the information about the current status of the experiment.

        Lists  all the parameters/arguments of the experiment and informs whether
        the experiment can be run.

        Args:
            info: Dictionary of sub-classes relevant attributes status that will be
                completed with some additional attributes status defined in this class.
                Defaults to None (no entries of sub-classes available or of importance).
            missing_object_to_check: dictionary mapping sub-classes attributes to
                attribute names, that may be needed to fully run the object. Defaults
                to None (no check will be performed).

        Returns:
            dictionary containing all pieces of information, with 2 entries:
                `Arguments` mapping a list of all argument, and `Values` mapping
                a list copntaining all the values.

        Raises:
            KeyError: if `Arguments` or `Values` entry is missing in passing argument `info`
        """
        # at this point all attributes are initialized (in constructor)
        if info is None:
            info = self._create_default_info_structure()
        info['Arguments'].extend([
            'Training Plan Class',
            'Model Arguments',
            'Training Arguments'
        ])
        info['Values'].extend(['\n'.join(findall('.{1,60}',
                                         str(e))) for e in [
            self._training_plan_class,
            self._model_args,
            self._training_args
        ]])

        return super().info(info, missing)

    def _check_missing_objects(self, missing_objects: Optional[Dict[Any, str]] = None) -> str:
        """Checks if some objects required for running the `run` method are not set"""
        # definitions of elements that are needed (paramount) for running the experiment
        _not_runnable_if_missing = {'Training Plan Class' : self._training_plan_class}

        missing: str = ''
        missing += super()._check_missing_objects(_not_runnable_if_missing)
        return missing


    @exp_exceptions
    def set_training_args(
        self,
        training_args: Union[dict, TrainingArgs, None]
    ) -> Union[dict, None]:
        """ Sets `training_args` + verification on arguments type

        Args:
            training_args: contains training arguments passed to the
                training plan's `training_routine` such as lr, epochs, batch_size...

        Returns:
            Training arguments

        Raises:
            FedbiomedExperimentError : bad training_args type
        """

        if isinstance(training_args, TrainingArgs):
            self._training_args = deepcopy(training_args)
        elif isinstance(training_args, dict) or training_args is None:
            self._training_args = TrainingArgs(training_args, only_required=False)
        else:
            msg = f"{ErrorNumbers.FB410.value} in function `set_training_args`. " \
                  "Expected type TrainingArgs, dict, or " \
                  f"None, got {type(training_args)} instead."
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._training_args.dict()



    @exp_exceptions
    def set_training_plan_class(self,
                                training_plan_class: Union[TrainingPlanT, None],
                                keep_weights: bool = True
                                ) -> Union[TrainingPlanT, None]:
        """Sets  the training plan type + verification on arguments type

        !!! warning "Resets the training plan"
            This function has an important (and intended!) side-effect: it resets the `training_plan` attribute.
            By default, it tries to keep the same weights as the current training plan, if available.

        Args:
            training_plan_class: training plan class to be used for training.
                For experiment to be properly and fully defined `training_plan_class` needs to be a `TrainingPlanT`
                Defaults to None (no training plan class defined yet)
            keep_weights: try to keep the same weights as the current training plan

        Returns:
            `training_plan_class` that is set for experiment

        Raises:
            FedbiomedExperimentError : bad training_plan_class type
        """
        if training_plan_class is None:
            self._training_plan_class = None
        elif inspect.isclass(training_plan_class):
            # training_plan_class must be a subclass of a valid training plan
            if issubclass(training_plan_class, TRAINING_PLAN_TYPES):
                # valid class
                self._training_plan_class = training_plan_class
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

        self._update_training_plan(keep_weights)  # resets the training plan attribute

        return self._training_plan_class

    @exp_exceptions
    def set_model_args(self,
                       model_args: dict,
                       keep_weights: bool = True) -> dict:
        """Sets `model_args` + verification on arguments type

        !!! warning "Resets the training plan"
            This function has an important (and intended!) side-effect: it resets the `training_plan` attribute.
            By default, it tries to keep the same weights as the current training plan, if available.

        Args:
            model_args (dict): contains model arguments passed to the constructor
                of the training plan when instantiating it : output and input feature
                dimension, etc.
            keep_weights: try to keep the same weights as the current training plan

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

        self._update_training_plan(keep_weights)  # resets the training plan attribute

        return self._model_args

    @exp_exceptions
    def check_training_plan_status(self) -> Dict:
        """ Method for checking training plan status, ie whether it is approved or not by the nodes

        Raises:
            FedbiomedExperimentError: if the training data is not defined.

        Returns:
            Training plan status for answering nodes
        """
        if self.training_data() is None:
            msg = f"{ErrorNumbers.FB410.value}. Cannot check training plan status: training data is not defined." \
                  f"Please either use the `set_tags` or `set_training_data` method to fix this."
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        job = TrainingPlanCheckJob(
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=self.training_data().node_ids(),
            keep_files_dir=self.experimentation_path(),
            experiment_id=self._experiment_id,
            training_plan=self.training_plan()
        )
        responses = job.execute()
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
            description: Description for training plan approve request

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the training plan has been correctly downloaded on the node side.
            Warning: status does not mean that the training plan is approved, only that it has been added
            to the "approval queue" on the node side.
        """
        job = TrainingPlanApproveJob(
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=self.training_data().node_ids(),
            keep_files_dir=self.experimentation_path(),
            training_plan=self.training_plan(),
            description=description,
        )
        responses = job.execute()
        return responses

    @exp_exceptions
    def breakpoint(self,
                   state,
                   bkpt_number) -> None:
        """
        Saves breakpoint with the state of the workflow.

        The following attributes will be saved:

          - training_args
          - training_plan_class
          - model_args
        """
        # save training plan to file
        training_plan_module = 'model_' + str(uuid.uuid4())
        training_plan_file = os.path.join(self.experimentation_path(), training_plan_module + '.py')
        self.training_plan().save_code(training_plan_file)

        state.update({
            'model_args': self._model_args,
            'training_plan_class_name': self._training_plan_class.__name__,
            'training_args': self._training_args.get_state_breakpoint(),
        })

        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(
                self.config.vars["EXPERIMENTS_DIR"],
                self._experimentation_folder,
                bkpt_number - 1
            )

        # rewrite paths in breakpoint : use the links in breakpoint directory
        state['training_plan_path'] = create_unique_link(
            breakpoint_path,
            # - Need a file with a restricted characters set in name to be able to import as module
            'model_' + str("{:04d}".format(bkpt_number - 1)), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('..', os.path.basename(training_plan_file))
        )
        params_path = os.path.join(breakpoint_path, f"model_params_{uuid.uuid4()}.mpk")
        Serializer.dump(self.training_plan().get_model_wrapper_class().get_weights(
            only_trainable = False, exclude_buffers = False), params_path)
        state['model_weights_path'] = params_path

        super().breakpoint(state, bkpt_number)


    @classmethod
    @exp_exceptions
    def load_breakpoint(cls,
                        breakpoint_folder_path: Optional[str] = None) -> Tuple[TrainingPlanWorkflowT, dict]:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so the workflow can be resumed.

        Args:
          breakpoint_folder_path: path of the breakpoint folder. Path can be absolute or relative eg:
            "var/experiments/Experiment_xxxx/breakpoints_xxxx". If None, loads the latest breakpoint of the latest
            workflow. Defaults to None.

        Returns:
            Reinitialized workflow object.

        Raises:
            FedbiomedExperimentError: bad argument type, error when reading breakpoint or bad loaded breakpoint
                content (corrupted)
        """
        loaded_exp, saved_state = super().load_breakpoint(breakpoint_folder_path)

        # Define type for pylint
        loaded_exp: TrainingPlanWorkflow

        # Import TP class
        _, tp_class = import_class_from_file(
            module_path=saved_state.get("training_plan_path"),
            class_name=saved_state.get("training_plan_class_name")
        )

        loaded_exp.set_model_args(saved_state["model_args"])
        loaded_exp.set_training_plan_class(tp_class)
        loaded_exp.set_training_args(
            TrainingArgs.load_state_breakpoint(
                saved_state.get('training_args')))
        training_plan = loaded_exp.training_plan()
        if training_plan is None:
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                'breakpoint file seems corrupted, `training_plan` is None'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        param_path = saved_state['model_weights_path']
        params = Serializer.load(param_path)
        loaded_exp.training_plan().get_model_wrapper_class().set_weights(params)

        return loaded_exp, saved_state

    def _check_round_value_consistency(
        self,
        round_current: int,
        variable_name: str
    ) -> bool:
        """Checks round value is consistant, ie it is a non negative integer.

        Args:
            round_current: Round to set
            variable_name: Argument name used for setting round.

        Raises:
            FedbiomedValueError: If round value is invalid
            FedbiomedTypeError: If round type is not correct
            """
        if not isinstance(round_current, int):
            msg = ErrorNumbers.FB410.value + f' `{variable_name}` of type : {type(round_current)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        if round_current < 0.:
            # cannot set a round <0
            msg = ErrorNumbers.FB410.value + f' `{variable_name}` cannot be negative or zero: {round_current}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return True

    @contextmanager
    def _keep_weights(self, keep_weights: bool):
        """Keeps same weights of training plan.

        Context manager for trying to keep the same weights as the
        current training plan after modifying it

        Args:
            keep_weights: If True tries to keep the weights as non-changed.
        """

        if keep_weights and self._training_plan is not None:
            weights = self._training_plan.get_model_params(exclude_buffers=False)
            yield
            try:
                self._training_plan.set_model_params(weights)
            except Exception as e:
                msg = f"{ErrorNumbers.FB410.value}. Attempting to keep same weights even though model has changed " \
                      f"failed with the following error message \n{e}\n Your model is now in an inconsistent state. " \
                      f"Try re-running the intended method by also setting `keep_weights=False` as parameter to " \
                      f"force resetting the model."
                raise FedbiomedExperimentError(msg)
        else:
            yield
