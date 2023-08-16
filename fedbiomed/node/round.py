# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
implementation of Round class of the node component
'''

import importlib
import inspect
import os
import sys
import time
import functools
import uuid
from typing import Dict, Union, Any, Optional, Tuple, List


from fedbiomed.common.constants import ErrorNumbers, TrainingPlanApprovalStatus
from fedbiomed.common.data import DataManager, DataLoadingPlan
from fedbiomed.common.exceptions import (
    FedbiomedError, FedbiomedOptimizerError, FedbiomedRoundError, FedbiomedUserInputError
)
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages
from fedbiomed.common.optimizers import BaseOptimizer, Optimizer
from fedbiomed.common.repository import Repository
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs

from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.secagg_manager import SKManager, BPrimeManager
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager
from fedbiomed.common.secagg import SecaggCrypter


class Round:
    """
    This class represents the training part execute by a node in a given round
    """

    def __init__(
        self,
        model_kwargs: Optional[dict] = None,
        training_kwargs: Optional[dict] = None,
        training: bool = True,
        dataset: Optional[dict] = None,
        training_plan_url: Optional[str] = None,
        training_plan_class: Optional[str] = None,
        params_url: Optional[str] = None,
        job_id: Optional[str] = None,
        researcher_id: Optional[str] = None,
        history_monitor: Optional[HistoryMonitor] = None,
        aggregator_args: Optional[Dict[str, Any]] = None,
        node_args: Union[dict, None] = None,
        round_number: int = 0,
        dlp_and_loading_block_metadata: Optional[Tuple[dict, List[dict]]] = None,
        aux_var_urls: Optional[List[str]] = None,
    ) -> None:
        """Constructor of the class

        Args:
            model_kwargs: contains model args. Defaults to None.
            training_kwargs: contains training arguments. Defaults to None.
            training: whether to perform a model training or just to perform a validation check (model infering)
            dataset: dataset details to use in this round. It contains the dataset name, dataset's id,
                data path, its shape, its description... . Defaults to None.
            training_plan_url: url from which to download training plan file
            training_plan_class: name of the training plan (eg 'MyTrainingPlan')
            params_url: url from which to upload/download model params
            job_id: job id
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
            aux_var_urls: Optional tuple of URLs of files containing Optimizer auxiliary variables
                that are to be processed by the training plan's optimizer.
        """

        self._use_secagg: bool = False
        self.dataset = dataset
        self.training_plan_url = training_plan_url
        self.training_plan_class = training_plan_class
        self.params_url = params_url
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.history_monitor = history_monitor
        self.aggregator_args = aggregator_args
        self.aux_var_urls = aux_var_urls or []

        self.tp_security_manager = TrainingPlanSecurityManager()
        self.node_args = node_args
        self.repository = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])
        self.training_plan = None
        self.training = training
        self._dlp_and_loading_block_metadata = dlp_and_loading_block_metadata

        self.training_kwargs = training_kwargs
        self.model_arguments = model_kwargs
        self.testing_arguments = None
        self.loader_arguments = None
        self.training_arguments = None
        self._secagg_crypter = SecaggCrypter()
        self._secagg_clipping_range = None
        self._round = round_number
        self._biprime = None
        self._servkey = None
        self._optim_aux_var = {}  # type: Dict[str, Dict[str, Any]]

    def initialize_validate_training_arguments(self) -> None:
        """Validates and separates training argument for experiment round"""

        self.training_arguments = TrainingArgs(self.training_kwargs, only_required=False)
        self.testing_arguments = self.training_arguments.testing_arguments()
        self.loader_arguments = self.training_arguments.loader_arguments()

    def download_aggregator_args(self) -> Tuple[bool, str]:
        """Retrieves aggregator arguments, that are sent through file exchange service

        Returns:
            Tuple[bool, str]: a tuple containing:
                a bool that indicates the success of operation
                a string containing the error message
        """
        # download heavy aggregator args (if any)

        if self.aggregator_args is not None:

            for arg_name, aggregator_arg in self.aggregator_args.items():
                if isinstance(aggregator_arg, dict):
                    url = aggregator_arg.get('url', False)

                    if any((url, arg_name)):
                        # if both `filename` and `arg_name` fields are present, it means that parameters
                        # should be retrieved using file
                        # exchanged system
                        success, param_path, error_msg = self.download_file(url, f"{arg_name}_{uuid.uuid4()}.mpk")
                        if not success:
                            return success, error_msg
                        else:
                            # FIXME: should we load parameters here or in the training plan
                            self.aggregator_args[arg_name] = {'param_path': param_path,
                                                              # 'params': training_plan.load(param_path,
                                                              # update_model=True)
                                                              }
                        self.aggregator_args[arg_name] = Serializer.load(param_path)
            return True, ''
        else:
            return True, "no file downloads required for aggregator args"

    def download_optimizer_aux_var(self) -> Tuple[bool, str]:
        """Download Optimizer auxiliary variables, if any.

        When this operation is successful, it updates a private attribute
        of this instance storing the obtained auxiliary variables (if any).

        Returns:
            success: Whether the operation was successful.
            error_message: Error message in case the operation failed.
        """
        aux_var = {}  # type: Dict[str, Dict[str, Any]]
        for url in self.aux_var_urls:
            # Download the file.
            path = f"aux_var_{uuid.uuid4()}.mpk"
            success, path, err_msg = self.download_file(url, path)
            if not success:
                error_message = (
                    f"Failed to download Optimizer aux var file: {err_msg}"
                )
                return False, error_message
            # Deserialize the file and add its contents to `aux_var`.
            try:
                aux_new = Serializer.load(path)
            except Exception as exc:
                error_message = (
                    f"Failed to de-serialize Optimizer aux var file {path}"
                    f": {repr(exc)}"
                )
                return False, error_message
            aux_var.update(aux_new)
        # If everything went fine, update private attribute and return flags.
        self._optim_aux_var = aux_var
        return True, ""

    def download_file(self, url: str, file_path: str) -> Tuple[bool, str, str]:
        """Downloads file from file exchange system

        Args:
            url (str): url used to download file
            file_path (str): file path used to store the downloaded content

        Returns:
            Tuple[bool, str, str]: tuple that contains:
                bool that indicates the success of the download
                str that returns the complete path file
                str containing the error message (if any). Returns empty
                string if operation successful.
        """
        status, params_path = self.repository.download_file(url, file_path)
        if (status != 200) or (params_path is None):
            error_message = f"Cannot download param file: {url}"
            return False, '', error_message
        return True, params_path, ''

    def _configure_secagg(
            self,
            secagg_servkey_id: Union[str, None] = None,
            secagg_biprime_id: Union[str, None] = None,
            secagg_random: Union[float, None] = None,
    ):
        """Validates secure aggregation status

        Args:
            secagg_servkey_id: Secure aggregation ID attached to the train request
            secagg_biprime_id: Secure aggregation Biprime context id that is going to be used for encryption
            secagg_random: Random number to validate encryption

        Returns:
            True if secure aggregation should be used.

        Raises:
            FedbiomedRoundError: incoherent secure aggregation status
        """

        secagg_all_none = all([s is None for s in (secagg_servkey_id, secagg_biprime_id)])
        secagg_all_defined = all([s is not None for s in (secagg_servkey_id, secagg_biprime_id)])

        if not secagg_all_none and not secagg_all_defined:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Missing secagg context. Please make sure that "
                                      f"train request contains both `secagg_servkey_id` and `secagg_biprime_id`.")

        if environ["FORCE_SECURE_AGGREGATION"] and secagg_all_none:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Node requires to apply secure aggregation but "
                                      f"Secure aggregation context for the training is not defined.")

        if secagg_all_defined and not environ["SECURE_AGGREGATION"]:
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value} Secure aggregation is not activated on the node."
            )

        if secagg_all_defined and secagg_random is None:
            raise FedbiomedRoundError(
                f"{ErrorNumbers.FB314.value} Secure aggregation requires to have random value to validate "
                f"secure aggregation correctness. Please add `secagg_random` to the train request"
            )

        if secagg_all_defined:
            self._biprime = BPrimeManager.get(secagg_id=secagg_biprime_id)
            self._servkey = SKManager.get(secagg_id=secagg_servkey_id, job_id=self.job_id)

            if self._biprime is None:
                raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Biprime for secagg: {secagg_biprime_id} "
                                          f"is not existing. Aborting train request.")

            if self._servkey is None:
                raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Server-key/user-key share for "
                                          f"secagg: {secagg_servkey_id} is not existing. "
                                          f"Aborting train request.")

        return secagg_all_defined

    def run_model_training(
            self,
            secagg_arguments: Union[Dict, None] = None,
    ) -> Dict[str, Any]:
        """This method downloads training plan file; then runs the training of a model
        and finally uploads model params to the file repository

        Args:
            secagg_arguments:
                - secagg_servkey_id: Secure aggregation Servkey context id. None means that the parameters
                    are not going to be encrypted
                - secagg_biprime_id: Secure aggregation Biprime context ID.
                - secagg_random: Float value to validate secure aggregation on the researcher side

        Returns:
            Returns the corresponding node message, training reply instance
        """
        # Validate secagg status. Raises error if the training request is compatible with
        # secure aggregation settings
        try:
            secagg_arguments = {} if secagg_arguments is None else secagg_arguments
            self._use_secagg = self._configure_secagg(
                secagg_servkey_id=secagg_arguments.get('secagg_servkey_id'),
                secagg_biprime_id=secagg_arguments.get('secagg_biprime_id'),
                secagg_random=secagg_arguments.get('secagg_random')
            )
        except FedbiomedRoundError as e:
            return self._send_round_reply(success=False, message=str(e))

        # Initialize and validate requested experiment/training arguments.
        try:
            self.initialize_validate_training_arguments()
        except FedbiomedUserInputError as e:
            return self._send_round_reply(success=False, message=repr(e))
        except Exception as e:
            msg = 'Unexpected error while validating training argument'
            logger.debug(f"{msg}: {repr(e)}")
            return self._send_round_reply(success=False, message=f'{msg}. Please contact system provider')

        # Download and validate the training plan.
        # Download the model weights and any auxiliary information.
        try:
            # module name cannot contain dashes
            import_module = 'training_plan_' + str(uuid.uuid4().hex)
            status, _ = self.repository.download_file(self.training_plan_url,
                                                      import_module + '.py')
            
            if status != 200:
                error_message = "Cannot download training plan file: " + self.training_plan_url
                return self._send_round_reply(success=False, message=error_message)
            else:
                if environ["TRAINING_PLAN_APPROVAL"]:
                    approved, training_plan_ = self.tp_security_manager.check_training_plan_status(
                        os.path.join(environ["TMP_DIR"], import_module + '.py'),
                        TrainingPlanApprovalStatus.APPROVED)

                    if not approved:
                        error_message = f'Requested training plan is not approved by the node: {environ["NODE_ID"]}'
                        return self._send_round_reply(success=False, message=error_message)
                    else:
                        logger.info(f'Training plan has been approved by the node {training_plan_["name"]}')

            # Retrieve model parameters and other auxiliary information.
            success, params_path, error_msg = self.download_file(
                self.params_url, f"my_model_{uuid.uuid4()}.mpk"
            )
            if success:
                success, error_msg = self.download_aggregator_args()
            if success:
                success, error_msg = self.download_optimizer_aux_var()
            if not success:
                return self._send_round_reply(success=False, message=error_msg)

        except Exception as e:
            # FIXME: this will trigger if model is not approved by node
            error_message = f"Cannot download training plan files: {repr(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # import module, declare the training plan, load parameters
        try:
            sys.path.insert(0, environ['TMP_DIR'])
            module = importlib.import_module(import_module)
            train_class = getattr(module, self.training_plan_class)
            self.training_plan = train_class()
            sys.path.pop(0)
        except Exception as e:
            error_message = f"Cannot instantiate training plan object: {repr(e)}"
            return self._send_round_reply(success=False, message=error_message)

        try:
            self.training_plan.post_init(model_args=self.model_arguments,
                                         training_args=self.training_arguments,
                                         aggregator_args=self.aggregator_args)
        except Exception as e:
            error_message = f"Can't initialize training plan with the arguments: {repr(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # import model params into the training plan instance
        try:
            params = Serializer.load(params_path)["model_weights"]
            self.training_plan.set_model_params(params)
        except Exception as e:
            error_message = f"Cannot initialize model parameters: {repr(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # Process Optimizer auxiliary variables, if any.
        error_message = self.process_optim_aux_var()
        if error_message:
            return self._send_round_reply(success=False, message=error_message)

        # Split training and validation data
        try:
            self._set_training_testing_data_loaders()
        except FedbiomedError as fe:
            error_message = f"Can not create validation/train data: {repr(fe)}"
            return self._send_round_reply(success=False, message=error_message)
        except Exception as e:
            error_message = f"Undetermined error while creating data for training/validation. Can not create " \
                            f"validation/train data: {repr(e)}"
            return self._send_round_reply(success=False, message=error_message)

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
                                 f"{repr(e)}")
                except Exception as e:
                    logger.error(f"Undetermined error during the testing phase on global parameter updates: "
                                 f"{repr(e)}")
            else:
                logger.error(f"{ErrorNumbers.FB314}: Can not execute validation routine due to missing testing dataset"
                             f"Please make sure that `test_ratio` has been set correctly")

        # If training is activated.
        if self.training:
            results = {}

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
                            f"{repr(e)}")
                    except Exception as e:
                        logger.error(f"Undetermined error during the validation phase on local parameter updates"
                                     f"{repr(e)}")
                else:
                    logger.error(
                        f"{ErrorNumbers.FB314.value}: Can not execute validation routine due to missing testing "
                        f"dataset please make sure that test_ratio has been set correctly")

            sample_size = len(self.training_plan.training_data_loader.dataset)

            results["encrypted"] = False
            model_weights = self.training_plan.after_training_params(flatten=self._use_secagg)
            if self._use_secagg:
                logger.info("Encrypting model parameters. This process can take some time depending on model size.")

                encrypt = functools.partial(
                    self._secagg_crypter.encrypt,
                    num_nodes=len(self._servkey["parties"]) - 1,  # -1: don't count researcher
                    current_round=self._round,
                    key=self._servkey["context"]["server_key"],
                    biprime=self._biprime["context"]["biprime"],
                    weight=sample_size,
                    clipping_range=secagg_arguments.get('secagg_clipping_range')
                )
                model_weights = encrypt(params=model_weights)
                results["encrypted"] = True
                results["encryption_factor"] = encrypt(params=[secagg_arguments["secagg_random"]])
                logger.info("Encryption is completed!")

            results['researcher_id'] = self.researcher_id
            results['job_id'] = self.job_id
            results['model_weights'] = model_weights
            results['node_id'] = environ['NODE_ID']
            results['optimizer_args'] = self.training_plan.optimizer_args()

            try:
                # TODO: add validation status to these results?
                # Dump the results to a msgpack file.
                filename = os.path.join(environ["TMP_DIR"], f"node_params_{uuid.uuid4()}.mpk")
                Serializer.dump(results, filename)
                # Upload that file to the remote repository.
                res = self.repository.upload_file(filename)
                logger.info("results uploaded successfully ")
            except Exception as exc:
                return self._send_round_reply(success=False, message=f"Cannot upload results: {exc}")

            # end : clean the namespace
            try:
                del self.training_plan
                del import_module
            except Exception as e:
                logger.debug(f'Exception raise while deleting training plan instance: {repr(e)}')

            return self._send_round_reply(success=True,
                                          timing={'rtime_training': rtime_after - rtime_before,
                                                  'ptime_training': ptime_after - ptime_before},
                                          params_url=res['file'],
                                          sample_size=sample_size)
        else:
            # Only for validation
            return self._send_round_reply(success=True)

    def _send_round_reply(
            self,
            message: str = '',
            success: bool = False,
            params_url: Union[str, None] = '',
            timing: dict = {},
            sample_size: Union[int, None] = None
    ) -> Dict[str, Any]:
        """
        Private method for sending reply to researcher after training/validation. Message content changes
        based on success status.

        Args:
            message: Message regarding the process.
            success: Declares whether training/validation is successful
            params_url: URL where parameters are uploaded
            timing: Timing statistics

        Returns:
            reply message
        """

        # If round is not successful log error message
        if not success:
            logger.error(message)

        return NodeMessages.format_outgoing_message({'node_id': environ['NODE_ID'],
                                          'job_id': self.job_id,
                                          'researcher_id': self.researcher_id,
                                          'command': 'train',
                                          'success': success,
                                          'dataset_id': self.dataset['dataset_id'] if success else '',
                                          'params_url': params_url,
                                          'msg': message,
                                          'sample_size': sample_size,
                                          'timing': timing}).get_dict()

    def process_optim_aux_var(self) -> str:
        """Process researcher-emitted Optimizer auxiliary variables, if any.

        Returns:
            Error message, empty if the operation was successful.
        """
        # Early-exit if there are no auxiliary variables to process.
        if not self._optim_aux_var:
            return ""
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
            optimizer.optimizer.set_aux(self._optim_aux_var)
        except FedbiomedOptimizerError as exc:
            return (
                "TrainingPlan Optimizer failed to ingest the provided "
                f"auxiliary variables: {repr(exc)}"
            )
        return ""

    def collect_optim_aux_var(self) -> Dict[str, Any]:
        """Collect auxiliary variables from the wrapped Optimizer, if any.

        If the TrainingPlan does not use a Fed-BioMed Optimizer, return an
        empty dict. If it does not hold any BaseOptimizer however, raise a
        FedbiomedRoundError.
        """
        optimizer = self._get_base_optimizer()
        if isinstance(optimizer.optimizer, Optimizer):
            aux_var = optimizer.optimizer.get_aux()

            if aux_var and self._use_secagg:
                # TODO: remove the following warning when secagg compatibility has been fixed
                # if secagg is used, raise a warning that encryption is not working with auxiliary variable
                logger.warning(f'Node {environ["NODE_ID"]} optimizer is sending auxiliary variables to the Researcher, but those are not encrypted with SecAgg.'
                               'Auxiliary Variables may contain sensitive information about the Nodes.' 
                               'This issue will be fixed in a future version of Fed-BioMed')
            return aux_var
        return {}

    def _get_base_optimizer(self) -> BaseOptimizer:
        """Return the training plan's BaseOptimizer, or raise a FedbiomedRoundError.

        This method is merely a failsafe for the case when the training plan's
        optimizer initialization step is malfunctioning, which should never
        happen, lest the end-user writes wrongful code.
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
        test_global_updates = self.testing_arguments.get('test_on_global_updates', False)
        test_local_updates = self.testing_arguments.get('test_on_local_updates', False)

        # Inform user about mismatch arguments settings
        if test_ratio != 0 and test_local_updates is False and test_global_updates is False:
            logger.warning("Validation will not be perform for the round, since there is no validation activated. "
                           "Please set `test_on_global_updates`, `test_on_local_updates`, or both in the "
                           "experiment.")

        if test_ratio == 0 and (test_local_updates is False or test_global_updates is False):
            logger.warning(
                'There is no validation activated for the round. Please set flag for `test_on_global_updates`'
                ', `test_on_local_updates`, or both. Splitting dataset for validation will be ignored')

        # Setting validation and train subsets based on test_ratio
        training_data_loader, testing_data_loader = self._split_train_and_test_data(test_ratio=test_ratio)

        # Set models validating and training parts for training plan
        self.training_plan.set_data_loaders(train_data_loader=training_data_loader,
                                            test_data_loader=testing_data_loader)

    def _split_train_and_test_data(self, test_ratio: float = 0):
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
        training_plan_type = self.training_plan.type()
        try:
            data_manager = self.training_plan.training_data()
        except TypeError as e:
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

        # Split dataset as train and test
        return data_manager.split(test_ratio=test_ratio)
