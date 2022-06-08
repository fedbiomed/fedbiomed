'''
implementation of Round class of the node component
'''

import os
import sys
import time
import inspect
from typing import Union, Any
import uuid

from fedbiomed.common.constants import ErrorNumbers, ModelApprovalStatus
from fedbiomed.common.data import DataManager
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedRoundError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages
from fedbiomed.common.repository import Repository

from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.model_manager import ModelManager


class Round:
    """
    This class represents the training part execute by a node in a given round
    """

    def __init__(self,
                 model_kwargs: dict = None,
                 training_kwargs: dict = None,
                 training: bool = True,
                 dataset: dict = None,
                 model_url: str = None,
                 model_class: str = None,
                 params_url: str = None,
                 job_id: str = None,
                 researcher_id: str = None,
                 history_monitor: HistoryMonitor = None,
                 node_args: Union[dict, None] = None):

        """Constructor of the class

        Args:
            model_kwargs: contains model args
            training_kwargs: contains model characteristics, especially input  dimension (key: 'in_features')
                and output dimension (key: 'out_features')
            dataset: dataset details to use in this round. It contains the dataset name, dataset's id,
                data path, its shape, its description...
            model_url: url from which to download model
            model_class: name of the training plan (eg 'MyTrainingPlan')
            params_url: url from which to upload/download model params
            job_id: job id
            researcher_id: researcher id
            history_monitor: Sends real-time feed-back to end-user during training
            node_args: command line arguments for node. Can include:
                - `gpu (bool)`: propose use a GPU device if any is available.
                - `gpu_num (Union[int, None])`: if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available.
                - `gpu_only (bool)`: force use of a GPU device if any available, even if researcher
                    doesn't request for using a GPU.
        """
        testing_args_keys = ('test_ratio', 'test_on_local_updates',
                             'test_on_global_updates', 'test_metric',
                             'test_metric_args')

        self.model_kwargs = model_kwargs
        # Split testing and training arguments

        self.testing_arguments = {}
        for arg in testing_args_keys:
            self.testing_arguments[arg] = training_kwargs.get(arg, None)
            training_kwargs.pop(arg, None)

        self.batch_size = training_kwargs.get('batch_size', 48)
        training_kwargs.pop('batch_size', None)

        # Set training arguments after removing testing arguments
        self.training_kwargs = training_kwargs

        self.dataset = dataset
        self.model_url = model_url
        self.model_class = model_class
        self.params_url = params_url
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.history_monitor = history_monitor
        self.model_manager = ModelManager()
        self.node_args = node_args
        self.repository = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])
        self.model = None
        self.training = training

    def run_model_training(self) -> dict[str, Any]:
        """This method downloads model file; then runs the training of a model
        and finally uploads model params

        Returns:
            Returns the corresponding node message, training reply instance
        """
        is_failed = False
        error_message = ''

        # Download model, training routine, execute it and return model results
        try:
            # module name cannot contain dashes
            import_module = 'my_model_' + str(uuid.uuid4().hex)
            status, _ = self.repository.download_file(self.model_url,
                                                      import_module + '.py')

            if status != 200:
                error_message = "Cannot download model file: " + self.model_url
                return self._send_round_reply(success=False, message=error_message)
            else:
                if environ["MODEL_APPROVAL"]:
                    approved, model = self.model_manager.check_model_status(os.path.join(environ["TMP_DIR"],
                                                                            import_module + '.py'),
                                                                            ModelApprovalStatus.APPROVED)
                    if not approved:
                        error_message = f'Requested model is not approved by the node: {environ["NODE_ID"]}'
                        return self._send_round_reply(success=False, message=error_message)
                        
                    else:
                        logger.info(f'Model has been approved by the node {model["name"]}')

            if not is_failed:
                status, params_path = self.repository.download_file(
                    self.params_url,
                    'my_model_' + str(uuid.uuid4()) + '.pt')
                if (status != 200) or params_path is None:
                    error_message = f"Cannot download param file: {self.params_url}"
                    return self._send_round_reply(success=False, message=error_message)

        except Exception as e:
            is_failed = True
            # FIXME: this will trigger if model is not approved by node
            error_message = f"Cannot download model files: {str(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # import module, declare the model, load parameters
        try:
            sys.path.insert(0, environ['TMP_DIR'])

            # import TrainingPlan created by Researcher on node
            exec('import ' + import_module, globals())
            sys.path.pop(0)

            # instantiate model as `train_class`
            train_class = eval(import_module + '.' + self.model_class)
            if self.model_kwargs is None or len(self.model_kwargs) == 0:
                # case where no args have been found (default)
                self.model = train_class()
            else:
                # case where args have been found  (and passed)
                self.model = train_class(self.model_kwargs)
        except Exception as e:
            error_message = f"Cannot instantiate model object: {str(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # import model params into the model instance
        try:
            self.model.load(params_path, to_params=False)
        except Exception as e:
            error_message = f"Cannot initialize model parameters: f{str(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # Run the training routine
        # Caution: always provide values for node-side arguments
        # (history_monitor, node_args) especially if they are security
        # related, to avoid overloading by malicious researcher.
        #
        # We want to have explicit message in case of overloading attempt
        # (and continue training) though by default it fails with
        # "dict() got multiple values for keyword argument"
        node_side_args = ['history_monitor', 'node_args']
        for arg in node_side_args:
            if arg in self.training_kwargs:
                del self.training_kwargs[arg]
                logger.warning(f'Researcher trying to set node-side training parameter {arg}. '
                               f' Maybe a malicious researcher attack.')

        # Split training and testing data
        try:
            self._set_training_testing_data_loaders()
        except FedbiomedError as e:
            error_message = f"Can not create test/train data: {str(e)}"
            return self._send_round_reply(success=False, message=error_message)
        except Exception as e:
            error_message = f"Undetermined error while creating data for training/test. Can not create " \
                            f"test/train data: {str(e)}"
            return self._send_round_reply(success=False, message=error_message)

        training_kwargs_with_history = dict(history_monitor=self.history_monitor,
                                            node_args=self.node_args,
                                            **self.training_kwargs)
        logger.info(f'training with arguments {training_kwargs_with_history}')

        # Testing Before Training
        if self.testing_arguments.get('test_on_global_updates', False) is not False:

            # Last control to make sure testing data loader is set.
            if self.model.testing_data_loader is not None:
                try:
                    self.model.testing_routine(metric=self.testing_arguments.get('test_metric', None),
                                               metric_args=self.testing_arguments.get('test_metric_args', {}),
                                               history_monitor=self.history_monitor,
                                               before_train=True)
                except FedbiomedError as e:
                    logger.error(f"{ErrorNumbers.FB314}: During the testing phase on global parameter updates; "
                                 f"{str(e)}")
                except Exception as e:
                    logger.error(f"Undetermined error during the testing phase on global parameter updates: "
                                 f"{e}")
            else:
                logger.error(f"{ErrorNumbers.FB314}: Can not execute testing routine due to missing testing dataset"
                             f"Please make sure that `test_ratio` has been set correctly")
        #
        # If training is activated.
        if self.training:
            if self.model.training_data_loader is not None:
                try:
                    results = {}
                    rtime_before = time.perf_counter()
                    ptime_before = time.process_time()
                    self.model.training_routine(**training_kwargs_with_history)
                    rtime_after = time.perf_counter()
                    ptime_after = time.process_time()
                except Exception as e:
                    error_message = f"Cannot train model in round: {str(e)}"
                    return self._send_round_reply(success=False, message=error_message)

            # Testing after training
            if self.testing_arguments.get('test_on_local_updates', False) is not False:

                if self.model.testing_data_loader is not None:
                    try:
                        self.model.testing_routine(metric=self.testing_arguments.get('test_metric', None),
                                                   metric_args=self.testing_arguments.get('test_metric_args', {}),
                                                   history_monitor=self.history_monitor,
                                                   before_train=False)
                    except FedbiomedError as e:
                        logger.error(
                            f"{ErrorNumbers.FB314.value}: During the testing phase on local parameter updates; "
                            f"{str(e)}")
                    except Exception as e:
                        logger.error(f"Undetermined error during the testing phase on local parameter updates"
                                     f"{e}")
                else:
                    logger.error(
                        f"{ErrorNumbers.FB314.value}: Can not execute testing routine due to missing testing "
                        f"dataset please make sure that test_ratio has been set correctly")

            # Upload results
            results['researcher_id'] = self.researcher_id
            results['job_id'] = self.job_id
            results['model_params'] = self.model.after_training_params()
            results['node_id'] = environ['NODE_ID']
            try:
                # TODO : should test status code but not yet returned
                # by upload_file
                filename = environ['TMP_DIR'] + '/node_params_' + str(uuid.uuid4()) + '.pt'
                self.model.save(filename, results)
                res = self.repository.upload_file(filename)
                logger.info("results uploaded successfully ")
            except Exception as e:
                is_failed = True
                error_message = f"Cannot upload results: {str(e)}"
                return self._send_round_reply(success=False, message=error_message)

            # end : clean the namespace
            try:
                del self.model
                del import_module
            except Exception as e:
                logger.debug(f'Exception raise while deleting model {e}')
                pass

            return self._send_round_reply(success=True,
                                          timing={'rtime_training': rtime_after - rtime_before,
                                                  'ptime_training': ptime_after - ptime_before},
                                          params_url=res['file'])
        else:
            # Only for testing
            return self._send_round_reply(success=True)

    def _send_round_reply(self,
                          message: str = '',
                          success: bool = False,
                          params_url: Union[str, None] = '',
                          timing: dict = {}):
        """
        Private method for sending reply to researcher after training/testing. Message content changes
        based on success status.

        Args:
            message: Message regarding the process.
            success: Declares whether training/testing is successful
            params_url: URL where parameters are uploaded
            timing: Timing statistics
        """

        # If round is not successful log error message
        if not success:
            logger.error(message)

        return NodeMessages.reply_create({'node_id': environ['NODE_ID'],
                                          'job_id': self.job_id,
                                          'researcher_id': self.researcher_id,
                                          'command': 'train',
                                          'success': success,
                                          'dataset_id': self.dataset['dataset_id'] if success else '',
                                          'params_url': params_url,
                                          'msg': message,
                                          'timing': timing}).get_dict()

    def _set_training_testing_data_loaders(self):
        """
        Method for setting training and testing data loaders based on the training and testing
        arguments.
        """

        # Set requested data path for model training and testing
        self.model.set_dataset_path(self.dataset['path'])

        # Get testing parameters
        test_ratio = self.testing_arguments.get('test_ratio', 0)
        test_global_updates = self.testing_arguments.get('test_on_global_updates', False)
        test_local_updates = self.testing_arguments.get('test_on_local_updates', False)

        # Inform user about mismatch arguments settings
        if test_ratio != 0 and test_local_updates is False and test_global_updates is False:
            logger.warning("Testing will not be perform for the round, since there is no test activated. "
                           "Please set `test_on_global_updates`, `test_on_local_updates`, or both in the "
                           "experiment.")

        if test_ratio == 0 and (test_local_updates is False or test_global_updates is False):
            logger.warning('There is no test activated for the round. Please set flag for `test_on_global_updates`'
                           ', `test_on_local_updates`, or both. Splitting dataset for testing will be ignored')

        # Setting test and train subsets based on test_ratio
        training_data_loader, testing_data_loader = self._split_train_and_test_data(test_ratio=test_ratio)
        # Set models testing and training parts for model
        self.model.set_data_loaders(train_data_loader=training_data_loader,
                                    test_data_loader=testing_data_loader)

    def _split_train_and_test_data(self, test_ratio: float = 0):
        """
        Method for splitting training and testing data based on training plan type. It sets
        `dataset_path` for model and calls `training_data` method of training plan.

        Args:
            test_ratio: The ratio that represent test partition. Default is 0, means that
                            all the samples will be used for training.

        Raises:

            FedbiomedRoundError: - When the method `training_data` of training plan
                                    has unsupported arguments.
                                 - Error while calling `training_data` method
                                 - If the return value of `training_data` is not an instance of
                                   `fedbiomed.common.data.DataManager`.
                                 - If `load` method of DataManager returns an error
        """

        training_plan_type = self.model.type()

        # Inspect the arguments of the method `training_data`, because this
        # part is defined by user might include invalid arguments
        parameters = inspect.signature(self.model.training_data).parameters
        args = list(parameters.keys())

        # Currently, training_data only accepts batch_size
        if len(args) > 1 and 'batch_size' not in args:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}, `the arguments of `training_data` of training "
                                      f"plan contains unsupported argument. ")

        # Check batch_size is one of the argument of training_data method
        # `batch_size` is in used for only TorchTrainingPlan. If it is based to
        # sklearn, it will raise argument error
        try:
            if 'batch_size' in args:
                data_manager = self.model.training_data(batch_size=self.batch_size)
            else:
                data_manager = self.model.training_data()
        except Exception as e:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}, `The method `training_data` of the "
                                      f"{str(training_plan_type.value)} has failed: {str(e)}")


        # Check whether training_data returns proper instance
        # it should be always Fed-BioMed DataManager
        if not isinstance(data_manager, DataManager):
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: The method `training_data` should return an "
                                      f"object instance of `fedbiomed.common.data.DataManager`, "
                                      f"not {type(data_manager)}")

        # Specific datamanager based on training plan
        try:
            # This data manager can be data manager for PyTorch or Sk-Learn
            data_manager.load(tp_type=training_plan_type)
        except FedbiomedError as e:
            raise FedbiomedRoundError(f"{ErrorNumbers.FB314.value}: Error while loading data manager; {str(e)}")

        # Get dataset property
        if hasattr(data_manager.dataset, "set_dataset_parameters"):
            dataset_parameters = self.dataset.get("dataset_parameters", {})
            data_manager.dataset.set_dataset_parameters(dataset_parameters)

        # All Framework based data managers have the same methods
        # If testing ratio is 0,
        # self.testing_data will be equal to None
        # self.testing_data will be equal to all samples
        # If testing ratio is 1,
        # self.testing_data will be equal to all samples
        # self.testing_data will be equal to None

        # Split dataset as train and test
        return data_manager.split(test_ratio=test_ratio)
