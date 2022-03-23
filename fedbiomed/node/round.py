'''
implementation of Round class of the node component
'''

import os
import sys
import time
import inspect
from typing import Union
import uuid

from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, TrainReply
from fedbiomed.common.repository import Repository
from fedbiomed.common.constants import ErrorNumbers, MetricTypes

from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.model_manager import ModelManager
from fedbiomed.common.data import DataManager
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedRoundError


class Round:
    """
    This class repesents the training part execute by a node in a given round
    """

    def __init__(self,
                 model_kwargs: dict = None,
                 training_kwargs: dict = None,
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
            - model_kwargs (dict): contains model args
            - training_kwargs (dict): contains model characteristics,
                especially input  dimension (key: 'in_features')
                and output dimension (key: 'out_features')
            - dataset ([dict]): dataset details to use in this round.
                It contains the dataset name, dataset's id,
                data path, its shape, its
                description...
            - model_url (str): url from which to download model
            - model_class (str): name of the training plan
                (eg 'MyTrainingPlan')
            - params_url (str): url from which to upload/dowload model params
            - job_id (str): job id
            - researcher_id (str): researcher id
            - history_monitor (HistoryMonitor)
            - node_args (Union[dict, None]): command line arguments for node. Can include:
                - gpu (bool): propose use a GPU device if any is available.
                - gpu_num (Union[int, None]): if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available.
                - gpu_only (bool): force use of a GPU device if any available, even if researcher
                    doesnt request for using a GPU.
        """
        self.model_kwargs = model_kwargs
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
        self.training_data_loader = None
        self.testing_data_loader = None

        self._default_batch_size = 48  # default bath size

    def run_model_training(self) -> TrainReply:
        """This method downloads model file; then runs the training of a model
        and finally uploads model params

        Returns:
            [NodeMessages]: returns the corresponding node message,
            trainReply instance
        """
        is_failed = False
        error_message = ''

        # Download model, training routine, execute it and return model results
        try:
            # module name cannot contain dashes
            import_module = 'my_model_' + str(uuid.uuid4().hex)
            status, _ = self.repository.download_file(self.model_url,
                                                      import_module + '.py')

            if (status != 200):
                is_failed = True
                error_message = "Cannot download model file: " + self.model_url
            else:
                if environ["MODEL_APPROVAL"]:
                    approved, model = self.model_manager.check_is_model_approved(os.path.join(environ["TMP_DIR"],
                                                                                              import_module + '.py'))
                    if not approved:
                        is_failed = True
                        error_message = f'Requested model is not approved by the node: {environ["NODE_ID"]}'
                    else:
                        logger.info(f'Model has been approved by the node {model["name"]}')

            if not is_failed:
                status, params_path = self.repository.download_file(
                    self.params_url,
                    'my_model_' + str(uuid.uuid4()) + '.pt')
                if (status != 200) or params_path is None:
                    is_failed = True
                    error_message = "Cannot download param file: " \
                                    + self.params_url

        except Exception as e:
            is_failed = True
            # FIXME: this will trigger if model is not approved by node
            error_message = "Cannot download model files:" + str(e)

        # import module, declare the model, load parameters
        if not is_failed:
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
                is_failed = True
                error_message = "Cannot instantiate model object: " + str(e)

        # import model params into the model instance
        if not is_failed:
            try:
                self.model.load(params_path, to_params=False)
            except Exception as e:
                is_failed = True
                error_message = "Cannot initialize model parameters:" + str(e)

        # Run the training routine
        if not is_failed:
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
        if not is_failed:
            try:
                # Setting test and train subsets based on test_ratio
                self._set_train_and_test_data(test_ratio=0.2)
                # Set models testing and training parts for model
                self.model.set_data_loaders(train_data_loader=self.training_data_loader,
                                            test_data_loader=self.testing_data_loader)
            except FedbiomedError as e:
                is_failed = True
                error_message = str(e)

        if not is_failed:
            training_kwargs_with_history = dict(history_monitor=self.history_monitor,
                                                node_args=self.node_args,
                                                **self.training_kwargs)
            logger.info(f'training with arguments {training_kwargs_with_history}')

        # Testing Before Training ------------------------------------------------------------------------------------
        if not is_failed:
            self.model.testing_routine(metric=MetricTypes.RECALL,
                                       history_monitor=self.history_monitor,
                                       before_train=True)
        # -----------------------------------------------------------------------------------------------------------

        if not is_failed:
            try:
                results = {}
                rtime_before = time.perf_counter()
                ptime_before = time.process_time()
                self.model.training_routine(**training_kwargs_with_history)
                rtime_after = time.perf_counter()
                ptime_after = time.process_time()
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model in round: " + str(e)

        # Testing after training ------------------------------------------------------------------------------------
        if not is_failed:
            self.model.testing_routine(metric=MetricTypes.ACCURACY,
                                       history_monitor=self.history_monitor,
                                       before_train=False)
        # -----------------------------------------------------------------------------------------------------------

        if not is_failed:
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
                error_message = "Cannot upload results: " + str(e)

        # end : clean the namespace
        try:
            del self.model
            del import_module
        except Exception:
            pass

        if not is_failed:
            return NodeMessages.reply_create({'node_id': environ['NODE_ID'],
                                              'job_id': self.job_id,
                                              'researcher_id': self.researcher_id,
                                              'command': 'train',
                                              'success': True,
                                              'dataset_id': self.dataset['dataset_id'],
                                              'params_url': res['file'],
                                              'msg': '',
                                              'timing': {
                                                  'rtime_training': rtime_after - rtime_before,
                                                  'ptime_training': ptime_after - ptime_before}
                                              }).get_dict()
        else:
            logger.error(error_message)
            return NodeMessages.reply_create({'node_id': environ['NODE_ID'],
                                              'job_id': self.job_id,
                                              'researcher_id': self.researcher_id,
                                              'command': 'train',
                                              'success': False,
                                              'dataset_id': '',
                                              'params_url': '',
                                              'msg': error_message,
                                              'timing': {}}).get_dict()

    def _set_train_and_test_data(self, test_ratio: float = 0):
        """
        Method for splitting training and testing data based on training plan type. It sets
        `dataset_path` for model and calls `training_data` method of training plan.

        Args:
            test_ratio (float) : The ratio that represent test partition. Default is 0, means that
                            all the samples will be used for training.

        Raises:

            FedbiomedRoundError: - When the method `training_data` of training plan
                                    has unsupported arguments.
                                 - Error while calling `training_data` method
                                 - If the return value of `training_data` is not an instance of
                                   `fedbiomed.common.data.DataManager`.
                                 - If `load` method of DataManager returns an error
        """

        # TODO: Discuss this part should this part be in the BaseTrainingPLan

        # Set requested data path for model training and testing
        self.model.set_dataset_path(self.dataset['path'])

        # Get batch size from training argument if it is not exist use default batch size
        batch_size = self.training_kwargs.get('batch_size', self._default_batch_size)

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
                data_manager = self.model.training_data(batch_size=batch_size)
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

        # All Framework based data managers have the same methods
        # If testing ratio is 0,
        # self.testing_data will be equal to None
        # self.testing_data will be equal to all samples
        # If testing ratio is 1,
        # self.testing_data will be equal to all samples
        # self.testing_data will be equal to None

        # Split dataset as train and test
        self.training_data_loader, self.testing_data_loader = data_manager.split(test_ratio=test_ratio)
        # If testing is inactive following method can be called to load all samples as train
        # self.train_data = sp.da_data_manager.load_all_samples()
