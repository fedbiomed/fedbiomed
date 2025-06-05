# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
implementation of Round class of the node component
'''

import tempfile
import shutil
import os
import time
import uuid
from typing import Dict, Union, Any, Optional, Tuple, List


from fedbiomed.common.constants import ErrorNumbers, TrainingPlanApprovalStatus
from fedbiomed.common.data import DataManager, DataLoadingPlan
from fedbiomed.common.exceptions import (
    FedbiomedError, FedbiomedOptimizerError, FedbiomedRoundError,
    FedbiomedUserInputError, FedbiomedSecureAggregationError
)
from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainReply
from fedbiomed.common.optimizers import (
    AuxVar,
    BaseOptimizer,
    EncryptedAuxVar,
    Optimizer,
    flatten_auxvar_for_secagg,
)
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common import utils

from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.node_state_manager import NodeStateManager, NodeStateFileName
from fedbiomed.node.secagg import SecaggRound
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager


class Round:
    """
    This class represents the training part execute by a node in a given round
    """

    def __init__(
        self,
        root_dir: str,
        db: str,
        node_id: str,
        training_plan: str,
        training_plan_class: str,
        model_kwargs: dict,
        training_kwargs: dict,
        training: bool ,
        dataset: dict,
        params: str,
        experiment_id: str,
        researcher_id: str,
        history_monitor: HistoryMonitor,
        aggregator_args: Dict[str, Any],
        node_args: Dict,
        tp_security_manager: TrainingPlanSecurityManager,
        round_number: int = 0,
        dlp_and_loading_block_metadata: Optional[Tuple[dict, List[dict]]] = None,
        aux_vars: Optional[Dict[str, AuxVar]] = None,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            db: Path to node database file.
            node_id: Node id
            training_plan: code of the training plan for this round
            training_plan_class: class name of the training plan
            model_kwargs: contains model args. Defaults to None.
            training_kwargs: contains training arguments. Defaults to None.
            training: whether to perform a model training or just to perform a validation check (model infering)
            dataset: dataset details to use in this round. It contains the dataset name, dataset's id,
                data path, its shape, its description... . Defaults to None.
            params: parameters of the model
            experiment_id: experiment id
            researcher_id: researcher id
            history_monitor: Sends real-time feed-back to end-user during training
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
            node_args: command line arguments for node. Can include:
                - `gpu (bool)`: propose use a GPU device if any is available.
                - `gpu_num (Union[int, None])`: if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available.
                - `gpu_only (bool)`: force use of a GPU device if any available, even if researcher
                    doesn't request for using a GPU.
            tp_security_manager: Training plan security manager instance.
            dlp_and_loading_block_metadata: Data loading plan to apply, or None if no DLP for this round.
            round_number: number of the iteration for this experiment
            aux_vars: Optional optimizer auxiliary variables.
        """
        self._node_id = node_id
        self._db = db
        self._dir = root_dir

        self.dataset = dataset
        self.training_plan_source = training_plan
        self.training_plan_class = training_plan_class
        self.params = params
        self.experiment_id = experiment_id
        self.researcher_id = researcher_id
        self.history_monitor = history_monitor
        self.aggregator_args = aggregator_args
        self.aux_vars = aux_vars or {}
        self.node_args = node_args
        self.training = training
        self._dlp_and_loading_block_metadata = dlp_and_loading_block_metadata
        self.training_kwargs = training_kwargs
        self.model_arguments = model_kwargs

        # Class attributes
        self.tp_security_manager = tp_security_manager
        self.training_plan = None
        self.testing_arguments = None
        self.loader_arguments = None
        self.training_arguments = None
        self._secure_aggregation = None
        self.is_test_data_shuffled: bool = False
        self._testing_indexes: Dict = {
            'testing_index': [],
            'training_index': [],
            'test_ratio': None
        }
        self._round = round_number
        self._node_state_manager: NodeStateManager = NodeStateManager(
            self._dir, self._node_id, self._db
        )
        self._temp_dir = tempfile.TemporaryDirectory()
        self._keep_files_dir = self._temp_dir.name

    def __del__(self):
        """Class destructor"""
        # remove temporary files directory
        self._temp_dir.cleanup()

    def _initialize_validate_training_arguments(self) -> Optional[Dict[str, Any]]:
        """Initialize and validate requested experiment/training arguments.

        Returns:
            A dictionary containing the error message if an error is triggered while parsing training and testing
            arguments, None otherwise.
        """
        try:
            self.training_arguments = TrainingArgs(self.training_kwargs, only_required=False)
            self.testing_arguments = self.training_arguments.testing_arguments()
            self.loader_arguments = self.training_arguments.loader_arguments()
        except FedbiomedUserInputError as e:
            return self._send_round_reply(success=False, message=repr(e))
        except Exception as e:
            msg = 'Unexpected error while validating training argument'
            logger.debug(f"{msg}: {repr(e)}")
            return self._send_round_reply(success=False, message=f'{msg}. Please contact system provider')

        return None

    def initialize_arguments(self,
                             previous_state_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Initializes arguments for training and testing and the NodeStateManager, the latter handling
        Node state loading and saving.

        Args:
            previous_state_id: previous Node state id. Defaults to None (which is the state_id default value for the first Round).

        Returns:
            A dictionary containing the error message if an error is triggered while parsing training and testing
            arguments, None otherwise.

        !!! "Note"
            If secure aggregation is activated, model weights will be encrypted as well as the
            optimizer's auxiliary variables (only if the optimizer used is a `DeclearnOptimizer`).
        """
        # initialize Node State Manager
        self._node_state_manager.initialize(previous_state_id=previous_state_id,
                                            testing=not self.training)
        return self._initialize_validate_training_arguments()

    def run_model_training(
        self,
        tp_approval: bool,
        secagg_insecure_validation: bool,
        secagg_active: bool,
        force_secagg: bool,
        secagg_arguments: Union[Dict[str, Any], None] = None,
    ) -> TrainReply:
        """Runs one round of model training

        Args:
            tp_approval: True if training plan approval by node is requested
            secagg_insecure_validation: True if (potentially insecure) consistency check is enabled
            secagg_active: True if secure aggregation is enabled on node
            force_secagg: True is secure aggregation is mandatory on node
            secagg_arguments: arguments for secure aggregation, some are specific to the scheme

        Returns:
            Returns the corresponding node message, training reply instance
        """
        # Validate secagg status. Raises error if the training request is not compatible with
        # secure aggregation settings

        try:
            self._secure_aggregation = SecaggRound(
                db=self._db,
                node_id=self._node_id,
                secagg_arguments=secagg_arguments,
                secagg_active=secagg_active,
                force_secagg=force_secagg,
                experiment_id=self.experiment_id
            )
        except FedbiomedSecureAggregationError as e:
            logger.error(str(e))
            return self._send_round_reply(
                success=False,
                message='Could not configure secure aggregation on node')

        # Validate and load training plan
        if tp_approval:
            approved, training_plan_ = self.tp_security_manager.\
                check_training_plan_status(
                    self.training_plan_source,
                    TrainingPlanApprovalStatus.APPROVED)

            if not approved:
                return self._send_round_reply(
                    False,
                    f'Requested training plan is not approved by the node: {self._node_id}')
            else:
                logger.info(f'Training plan has been approved by the node {training_plan_["name"]}',
                            researcher_id=self.researcher_id)

        # Import training plan, save to file, reload, instantiate a training plan
        try:
            CurrentTPModule, CurrentTrainingPlan = utils.import_class_from_spec(
                code=self.training_plan_source, class_name=self.training_plan_class)
            self.training_plan = CurrentTrainingPlan()
        except Exception:
            error_message = "Cannot instantiate training plan object."
            return self._send_round_reply(success=False, message=error_message)

        # save and load training plan to a file to be sure
        # 1. a file is associated to training plan so we can read its source, etc.
        # 2. all dependencies are applied
        training_plan_module = 'model_' + str(uuid.uuid4())
        training_plan_file = os.path.join(self._keep_files_dir, training_plan_module + '.py')
        try:
            self.training_plan.save_code(training_plan_file, from_code=self.training_plan_source)
        except Exception as e:
            error_message = "Cannot save the training plan to a local tmp dir"
            logger.error(f"Cannot save the training plan to a local tmp dir : {e}")
            return self._send_round_reply(success=False, message=error_message)

        del CurrentTrainingPlan
        del CurrentTPModule

        try:
            CurrentTPModule, self.training_plan = utils.import_class_object_from_file(
                training_plan_file, self.training_plan_class)
        except Exception:
            error_message = "Cannot load training plan object from file."
            return self._send_round_reply(success=False, message=error_message)

        try:
            self.training_plan.post_init(model_args=self.model_arguments,
                                         training_args=self.training_arguments,
                                         aggregator_args=self.aggregator_args)
        except Exception:
            error_message = "Can't initialize training plan with the arguments."
            return self._send_round_reply(success=False, message=error_message)

        # load node state
        previous_state_id = self._node_state_manager.previous_state_id
        if previous_state_id is not None:
            try:
                self._load_round_state(previous_state_id)
            except Exception:
                # don't send error details
                return self._send_round_reply(success=False, message="Can't read previous node state.")

        # Load model parameters received from researcher
        try:
            self.training_plan.set_model_params(self.params)
        except Exception:
            error_message = "Cannot initialize model parameters."
            return self._send_round_reply(success=False, message=error_message)
        # ---------------------------------------------------------------------

        # Process Optimizer auxiliary variables, if any.
        error_message = self.process_optim_aux_var()
        if error_message:
            return self._send_round_reply(success=False, message=error_message)

        # Split training and validation data -------------------------------------
        try:

            self._set_training_testing_data_loaders()
            
        except FedbiomedError as fe:
            error_message = f"Can not create validation/train data: {repr(fe)}"
            return self._send_round_reply(success=False, message=error_message)
        except Exception as e:
            error_message = f"Undetermined error while creating data for training/validation. Can not create " \
                            f"validation/train data: {repr(e)}"
            return self._send_round_reply(success=False, message=error_message)
        # ------------------------------------------------------------------------


        # Validation Before Training
        if self.testing_arguments.get('test_on_global_updates', False) is not False:

            # Last control to make sure validation data loader is set.
            if self.training_plan.testing_data_loader is not None:
                try:
                    self.training_plan.testing_routine(metric=self.testing_arguments.get('test_metric', None),
                                                       metric_args=self.testing_arguments.get('test_metric_args', {}),
                                                       history_monitor=self.history_monitor,
                                                       before_train=True)
                except FedbiomedError as e:
                    logger.error(f"{ErrorNumbers.FB314}: During the validation phase on global parameter updates; "
                                 f"{repr(e)}", researcher_id=self.researcher_id)
                except Exception as e:
                    logger.error(f"Undetermined error during the testing phase on global parameter updates: "
                                 f"{repr(e)}", researcher_id=self.researcher_id)
            else:
                logger.error(f"{ErrorNumbers.FB314}: Can not execute validation routine due to missing testing dataset"
                             f"Please make sure that `test_ratio` has been set correctly",
                             researcher_id=self.researcher_id)

        # If training is activated.
        if self.training:
            results = {}  # type: Dict[str, Any]

            # Perform the training round.
            if self.training_plan.training_data_loader is not None:
                try:
                    rtime_before = time.perf_counter()
                    ptime_before = time.process_time()
                    self.training_plan.training_routine(history_monitor=self.history_monitor,
                                                        node_args=self.node_args)
                    rtime_after = time.perf_counter()
                    ptime_after = time.process_time()
                except Exception as exc:
                    error_message = f"Cannot train model in round: {repr(exc)}"
                    return self._send_round_reply(success=False, message=error_message)

            # Collect Optimizer auxiliary variables, if any.

            try:
                results['optim_aux_var'] = self.collect_optim_aux_var()
            except (FedbiomedOptimizerError, FedbiomedRoundError) as exc:
                error_message = f"Cannot collect Optimizer auxiliary variables: {repr(exc)}"
                return self._send_round_reply(success=False, message=error_message)

            # Validation after training
            if self.testing_arguments.get('test_on_local_updates', False) is not False:

                if self.training_plan.testing_data_loader is not None:
                    try:
                        self.training_plan.testing_routine(metric=self.testing_arguments.get('test_metric', None),
                                                           metric_args=self.testing_arguments.get('test_metric_args',
                                                                                                  {}),
                                                           history_monitor=self.history_monitor,
                                                           before_train=False)
                    except FedbiomedError as e:
                        logger.error(
                            f"{ErrorNumbers.FB314.value}: During the validation phase on local parameter updates; "
                            f"{repr(e)}", researcher_id=self.researcher_id)
                    except Exception as e:
                        logger.error(f"Undetermined error during the validation phase on local parameter updates"
                                     f"{repr(e)}", researcher_id=self.researcher_id)
                else:
                    logger.error(
                        f"{ErrorNumbers.FB314.value}: Can not execute validation routine due to missing testing "
                        f"dataset please make sure that test_ratio has been set correctly",
                        researcher_id=self.researcher_id)

            # FIXME: this will fail if `self.training_plan.training_data_loader = None` (see issue )
            results["sample_size"] = len(self.training_plan.training_data_loader.dataset)

            results["encrypted"] = False
            model_weights = self.training_plan.after_training_params(flatten=self._secure_aggregation.use_secagg)

            if self._secure_aggregation.use_secagg:
                model_weights, enc_factor, aux_var = self._encrypt_weights_and_auxvar(
                    model_weights=model_weights,
                    optim_aux_var=results["optim_aux_var"],
                    sample_size=results["sample_size"],
                    secagg_insecure_validation=secagg_insecure_validation,
                )
                results["encrypted"] = True
                results["encryption_factor"] = enc_factor
                if aux_var is not None:
                    results["optim_aux_var"] = aux_var.to_dict()
            results['params'] = model_weights
            results['optimizer_args'] = self.training_plan.optimizer_args()
            results['state_id'] = self._node_state_manager.state_id

            try:
                self._save_round_state()
            except Exception:
                # don't send details to researcher
                return self._send_round_reply(success=False, message="Can't save new node state.")

            # end : clean the namespace
            try:
                del self.training_plan
                del CurrentTPModule
            except Exception:
                logger.debug('Exception raised while deleting training plan instance')

            return self._send_round_reply(success=True,
                                          timing={'rtime_training': rtime_after - rtime_before,
                                                  'ptime_training': ptime_after - ptime_before},
                                          extend_with=results)
        else:
            # Only for validation
            return self._send_round_reply(success=True)

    def _encrypt_weights_and_auxvar(
        self,
        model_weights: List[float],
        optim_aux_var: Dict[str, AuxVar],
        sample_size: int,
        secagg_insecure_validation: bool,
    ) -> Tuple[List[int], List[int], Optional[EncryptedAuxVar]]:
        """Encrypt model weights and (opt.) optimizer auxiliary variables.

        Args:
            model_weights: Flattened model parameters to encrypt.
            optim_aux_var: Optional optimizer auxiliary variables to encrypt.
            sample_size: Number of training samples (used to weight model
                parameters).
            secagg_insecure_validation: True if (potentially insecure) consistency check is enabled

        Returns:
            encrypted_weights: Encrypted model parameters, as a list of int.
            encryption_factor: Encryptiong factor (based on a secagg argument).
            encrypted_aux_var: Optional `EncryptedAuxVar` instance storing
                encrypted optimizer auxiliary variables, if any.
        """
        # Case when optimizer auxiliary variables are to be encrypted.
        # TODO; find a way to encrypt safely aux var with model weights at once. See #1250.
        if optim_aux_var:
            logger.info(
                "Encrypting model parameters and optimizer auxiliary variables."
                "This process can take some time depending on model size.",
                researcher_id=self.researcher_id,
            )
            # Flatten optimizer auxiliary variables and divide them by scaling weights.
            cryptable, enc_specs, cleartext, clear_cls = (
                flatten_auxvar_for_secagg(optim_aux_var)
            )
            #cryptable = [x / sample_size for x in cryptable] # ?? already done while encrypting
            # Encrypt both model parameters and optimizer aux var at once. -> NO
            encrypted_aux = self._secure_aggregation.scheme.encrypt(
                            params=cryptable,
                            current_round=self._round,
                            weight=sample_size,
            )
            encrypted_aux = EncryptedAuxVar(
                encrypted=[encrypted_aux],
                enc_specs=enc_specs,
                cleartext=cleartext,
                clear_cls=clear_cls,
            )

        # Case when there are only model parameters to encrypt.
        else:
            logger.info(
                "Encrypting model parameters."
                "This process can take some time depending on model size.",
                researcher_id=self.researcher_id,
            )
            # encrypted_wgt = self._secure_aggregation.scheme.encrypt(
            #         params=model_weights,
            #         current_round=self._round,
            #         weight=sample_size,
            # )
            encrypted_aux = None

        encrypted_wgt = self._secure_aggregation.scheme.encrypt(
                            params=model_weights,
                            current_round=self._round,
                            weight=sample_size,
                    )

        encrypted_rng = None
        # At any rate, produce encryption factors.
        if self._secure_aggregation.scheme.secagg_random is not None and \
                secagg_insecure_validation:
            encrypted_rng = self._secure_aggregation.scheme.encrypt(
                        params=[self._secure_aggregation.scheme.secagg_random],
                        current_round=self._round,
                        weight=sample_size)

        logger.info("Encryption was completed!", researcher_id=self.researcher_id)

        return encrypted_wgt, encrypted_rng, encrypted_aux

    def _send_round_reply(
        self,
        success: bool = False,
        message: str = '',
        extend_with: Optional[Dict] = None,
        timing: dict = {},
    ) -> TrainReply:
        """Sends reply to researcher after training/validation.

        Message content changes based on success status.

        Args:
            success: Declares whether training/validation is successful
            message: Message regarding the process.
            extend_with: Extends the train reply
            timing: Timing statistics
        """

        if extend_with is None:
            extend_with = {}

        # If round is not successful log error message
        return TrainReply(**{
            'node_id': self._node_id,
            'experiment_id': self.experiment_id,
            'state_id': self._node_state_manager.state_id,
            'researcher_id': self.researcher_id,
            'success': success,
            'dataset_id': self.dataset['dataset_id'] if success else '',
            'msg': message,
            'timing': timing,
            **extend_with}
        )


    def process_optim_aux_var(self) -> Optional[str]:
        """Process researcher-emitted Optimizer auxiliary variables, if any.

        Returns:
            Error message, empty if the operation was successful.
        """
        # Early-exit if there are no auxiliary variables to process.
        if not any(self.aux_vars):
            return None
        # Fetch the training plan's BaseOptimizer.
        try:
            optimizer = self._get_base_optimizer()
        except FedbiomedRoundError as exc:
            return str(exc)
        # Verify that the BaseOptimizer wraps an Optimizer.
        if not isinstance(optimizer.optimizer, Optimizer):
            return (
                "Received Optimizer auxiliary variables, but the "
                "TrainingPlan does not manage a compatible Optimizer."
            )
        # Pass auxiliary variables to the Optimizer.
        try:
            optimizer.optimizer.set_aux(self.aux_vars)
        except FedbiomedOptimizerError as exc:
            return (
                "TrainingPlan Optimizer failed to ingest the provided "
                f"auxiliary variables: {repr(exc)}"
            )
        # early stop if secagg is activated and optimizer has more than one module that accepts
        # auxiliary variable
        if optimizer.count_nb_auxvar() > 1 and self._secure_aggregation.use_secagg:
            return (
                "Can not parse more than one `declearn` module requiring auxiliary variables while"
                " Secure Aggregation activated. Aborting..."
            )
        return None

    def _load_round_state(self, state_id: str) -> None:
        """Loads optimizer state of previous `Round`, given a `state_id`.

        Loads optimizer with default values if optimizer entry has not been found
        or if Optimizer type has changed between current and previous `Round`. Should
        be called at the begining of a `Round`, before training a model.
        If loading fails, skip the loading part and loads `Optimizer` with default values.

        Args:
            state_id: state_id from which to recover `Node`'s state

        Raises:
            FedbiomedRoundError: raised if `Round` doesnot have any `experiment_id` attribute.

        Returns:
            True
        """

        # define here all the object that should be reloaded from the node state database
        state = self._node_state_manager.get(self.experiment_id, state_id)

        optimizer_wrapper = self._get_base_optimizer()  # optimizer from TrainingPlan
        if state['optimizer_state'] is not None and \
           str(optimizer_wrapper.__class__) == state['optimizer_state']['optimizer_type']:

            optim_state_path = state['optimizer_state'].get('state_path')
            try:
                optim_state = Serializer.load(optim_state_path)

                optimizer_wrapper.load_state(optim_state, load_from_state=True)
                logger.debug(f"Optimizer loaded state {optim_state}")
                logger.info(f"State {state_id} loaded")

            except Exception as err:
                logger.warning(f"Loading Optimizer from state {state_id} failed ... Resuming Experiment with default"
                               "Optimizer state.")
                logger.debug(f" Error detail {err}")

        # load testing dataset if any
        if state['testing_dataset'] and not self.is_test_data_shuffled:
            self._testing_indexes = state['testing_dataset']

        # add below other components that need to be reloaded from node state database


    def _save_round_state(self) -> Dict:
        """Saves `Round` state (mainly Optimizer state) in database through
        [`NodeStateManager`][fedbiomed.node.node_state_manager.NodeStateManager].

        Some piece of information such as Optimizer state are also aved in files (located under
        <fedbiomed-node>/var/node_state<node_id>/experiment_id_<experiment_id>/).
        Should be called at the end of a `Round`, once the model has been trained.

        Entries saved in State:
        - optimizer_state:
            - optimizer_type (str)
            - state_path (str)

        Returns:
            `Round` state that will be saved in the database.
        """
        state: Dict[str, Any] = {}
        _success: bool = True

        # saving optimizer state
        optimizer = self._get_base_optimizer()

        optimizer_state = optimizer.save_state()
        if optimizer_state is not None:
            # this condition was made so we dont save stateless optimizers
            optim_path = self._node_state_manager.generate_folder_and_create_file_name(
                self.experiment_id,
                self._round,
                NodeStateFileName.OPTIMIZER
            )
            Serializer.dump(optimizer_state, path=optim_path)
            logger.debug("Saving optim state")

            optimizer_state_entry: Dict = {
                'optimizer_type': str(optimizer.__class__),
                'state_path': optim_path
            }
            # FIXME: we do not save auxiliary variables for scaffold, but not sure about what to do

        else:
            logger.warning(f"Unable to save optimizer state of type {type(optimizer)}. Skipping...")
            _success = False
            optimizer_state_entry = None
        state['optimizer_state'] = optimizer_state_entry

        # save testing dataset
        state['testing_dataset'] = None

        test_ratio = self._testing_indexes.get('test_ratio')
        test_ratio = test_ratio if not self.testing_arguments else self.testing_arguments.get('test_ratio', None)
        if not self.is_test_data_shuffled and test_ratio:
            self._testing_indexes['test_ratio'] = test_ratio
            state['testing_dataset'] = self._testing_indexes
            logger.info("testing dataset saved in database")
        else:
            logger.info("testing data will be reshuffled next rounds")
        # add here other object states (ie model state, ...)

        # save completed node state

        self._node_state_manager.add(self.experiment_id, state)
        if _success:
            logger.debug("Node state saved into DataBase")
        else:
            logger.debug("Node state has been partially saved into the Database")

        return state

    def collect_optim_aux_var(
        self,
    ) -> Dict[str, AuxVar]:
        """Collect auxiliary variables from the wrapped Optimizer, if any.

        If the TrainingPlan does not use a Fed-BioMed Optimizer, return an
        empty dict. If it does not hold any BaseOptimizer however, raise a
        FedbiomedRoundError.

        Returns:
            Auxiliary variables, as a `{module_name: module_auxvar}` dict.
        """
        optimizer = self._get_base_optimizer()
        if isinstance(optimizer.optimizer, Optimizer):
            return optimizer.optimizer.get_aux()
        return {}

    def _get_base_optimizer(self) -> BaseOptimizer:
        """Return the training plan's BaseOptimizer, or raise a FedbiomedRoundError.

        This method is merely a failsafe for the case when the training plan's
        optimizer initialization step is malfunctioning, which should never
        happen, lest the end-user writes wrongful code.

        Returns:
            Optimizer defined in training plan
        """
        optimizer = self.training_plan.optimizer()
        if optimizer is None:
            raise FedbiomedRoundError(
                "The TrainingPlan does not hold a BaseOptimizer after "
                "being initialized."
            )
        return optimizer

    def _set_training_testing_data_loaders(self):
        """
        Method for setting training and validation data loaders based on the training and validation
        arguments.
        """

        # Set requested data path for model training and validation
        self.training_plan.set_dataset_path(self.dataset['path'])

        # Get validation parameters
        test_ratio = self.testing_arguments.get('test_ratio', 0)
        self.is_test_data_shuffled = self.testing_arguments.get('shuffle_testing_dataset', False)
        test_global_updates = self.testing_arguments.get('test_on_global_updates', False)
        test_local_updates = self.testing_arguments.get('test_on_local_updates', False)

        # Inform user about mismatch arguments settings
        if test_ratio != 0 and test_local_updates is False and test_global_updates is False:
            logger.warning("Validation will not be performed for the round, since there is no validation activated. "
                           "Please set `test_on_global_updates`, `test_on_local_updates`, or both in the "
                           "experiment.",
                           researcher_id=self.researcher_id)

        if test_ratio == 0 and (test_local_updates is True or test_global_updates is True):
            logger.warning(
                'Validation is activated but `test_ratio` is 0. Please change `test_ratio`. '
                'No validation will be performed. Splitting dataset for validation will be ignored',
                researcher_id=self.researcher_id
            )

        # Setting validation and train subsets based on test_ratio
        training_data_loader, testing_data_loader = self._split_train_and_test_data(
                test_ratio=test_ratio,
                #random_seed=rand_seed
            )
        # Set models validating and training parts for training plan
        self.training_plan.set_data_loaders(train_data_loader=training_data_loader,
                                            test_data_loader=testing_data_loader)

    def _split_train_and_test_data(self, test_ratio: float = 0) -> DataManager:
        # FIXME: incorrect type output
        """
        Method for splitting training and validation data based on training plan type. It sets
        `dataset_path` for training plan and calls `training_data` method of training plan.

        Args:
            test_ratio: The ratio that represent validating partition. Default is 0, means that
                            all the samples will be used for training.

        Raises:

            FedbiomedRoundError: - When the method `training_data` of training plan
                                    has unsupported arguments.
                                 - Error while calling `training_data` method
                                 - If the return value of `training_data` is not an instance of
                                   `fedbiomed.common.data.DataManager`.
                                 - If `load` method of DataManager returns an error
        """

        training_plan_type = self.training_plan.type()  # FIXME: type is not part of the BaseTrainingPlan API
        try:
            data_manager = self.training_plan.training_data()
        except TypeError as e:
            # FIXME; TypeError could occur whithin the training_data method.
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}, `The method `training_data` of the "
                                      f"{str(training_plan_type)} should not take any arguments."
                                      f"Instead, the following error occurred: {repr(e)}")
        except Exception as e:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}, `The method `training_data` of the "
                                      f"{str(training_plan_type)} has failed: {repr(e)}")

        # Check whether training_data returns proper instance
        # it should be always Fed-BioMed DataManager
        if not isinstance(data_manager, DataManager):
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: The method `training_data` should return an "
                                      f"object instance of `fedbiomed.common.data.DataManager`, "
                                      f"not {type(data_manager)}")

        # Set loader arguments
        data_manager.extend_loader_args(self.loader_arguments)

        # Specific datamanager based on training plan
        try:
            # This data manager can be data manager for PyTorch or Sk-Learn
            data_manager.load(tp_type=training_plan_type)
        except FedbiomedError as e:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Error while loading data manager; {repr(e)}")
        # Get dataset property
        if hasattr(data_manager.dataset, "set_dataset_parameters"):
            dataset_parameters = self.dataset.get("dataset_parameters", {})
            data_manager.dataset.set_dataset_parameters(dataset_parameters)

        if self._dlp_and_loading_block_metadata is not None:
            if hasattr(data_manager.dataset, 'set_dlp'):
                dlp = DataLoadingPlan().deserialize(*self._dlp_and_loading_block_metadata)
                data_manager.dataset.set_dlp(dlp)
            else:
                raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Attempting to set DataLoadingPlan "
                                          f"{self._dlp_and_loading_block_metadata['name']} on dataset of type "
                                          f"{data_manager.dataset.__class__.__name__} which is not enabled.")

        # All Framework based data managers have the same methods
        # If testing ratio is 0,
        # self.testing_data will be equal to None
        # self.training_data will be equal to all samples
        # If testing ratio is 1,
        # self.testing_data will be equal to all samples
        # self.training_data will be equal to None

        # setting testing_index (if any)
        data_manager.load_state(self._testing_indexes)

        # Split dataset as train and test

        training_loader, testing_loader = data_manager.split(
            test_ratio=test_ratio,
            test_batch_size=self.testing_arguments.get('test_batch_size'),
            is_shuffled_testing_dataset = self.is_test_data_shuffled
        )
        # retrieve testing/training indexes
        self._testing_indexes = data_manager.save_state()

        return training_loader, testing_loader
