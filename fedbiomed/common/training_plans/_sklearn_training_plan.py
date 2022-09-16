"""TrainingPlan definitions for sklearn ML framework

This module implements the base class for all implementations of Fed-BioMed wrappers around scikit-learn models.
"""

import sys
import numpy as np

from typing import Any, Dict, Union, Callable, List, Tuple
from io import StringIO
from joblib import dump, load

from ._base_training_plan import BaseTrainingPlan

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans, ProcessTypes
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.utils import get_method_spec


class _Capturer(list):
    """Capturing class for the console output of the scikit-learn models during training
    when verbose is set to true.
    """

    def __enter__(self):
        sys.stdout.flush()
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # Remove it from memory
        sys.stdout = self._stdout


class SKLearnTrainingPlan(BaseTrainingPlan):
    """Base class for Fed-BioMed wrappers of sklearn classes.

    Classes that inherit from this must meet the following conditions:
    - have a `model` attribute with an instance of the scikit-learn class being wrapped
    - populate a `params_list` attribute during initialization with the model parameters to be used for aggregation
    - implement a `training_routine_hook` method that:
        1. sets `data` and `target` attributes as outputs of a data loader
        2. calls `partial_fit` or a similar method of the wrapped scikit-learn model
    - implement a `evaluate_loss` method that calls the `_evaluate_loss_core` method of this class (i.e. the base)

    Attributes:
        params: parameters of the model, both learnable and non-learnable
        model_args: model arguments provided by researcher
        param_list: names of the parameters that will be used for aggregation
        dataset_path: the path to the dataset on the node
    """

    def __init__(self):
        """
        Class initializer.

        Args:
        - model_args (dict, optional): model arguments. Defaults to {}.
        """
        super().__init__()
        self._model_args = None
        self._training_args = None
        self._params = None

        if getattr(self, '_model') is None:
            msg = ErrorNumbers.FB303.value + ": SKLEARN model is None"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        self._param_list = []
        self.__type = TrainingPlans.SkLearnTrainingPlan
        self._is_classification = False
        self._is_regression = False
        self._is_clustering = False
        self._is_binary_classification = False
        self._verbose_capture_option = False
        self.dataset_path = None
        self.add_dependency(["import inspect",
                             "import numpy as np",
                             "import pandas as pd",
                             "from fedbiomed.common.training_plans import SKLearnTrainingPlan",
                             "from fedbiomed.common.data import DataManager",
                             ])

    def post_init(self, model_args: Dict, training_args: Dict, optimizer_args: Dict) -> None:
        """ Instantiates model, training and optimizer arguments

        Args:
            model_args: Model arguments
            training_args: Training arguments
            optimizer_args: Optimizer arguments. Optimizer arguments are not make sense for SkLearn based training
                arguments. However, it is mandatory for `post_init` class in Round class
                because there is single round for SkLearn adn Torch.

        """
        dependencies: Union[Tuple, List] = self.init_dependencies()
        if not isinstance(dependencies, (list, tuple)):
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605}: Expected dependencies are l"
                                             f"ist or tuple, but got {type(dependencies)}")
        self.add_dependency(dependencies)

        self._model_args = model_args
        self._training_args = training_args
        self._params = self._model.get_params()
        self._params.update({key: self._model_args[key] for key in model_args if key in self._params})
        self.set_init_params()

    def model_args(self) -> Dict:
        """Retrieves model arguments

        Returns:
            Model arguments
        """
        return self._model_args

    def training_args(self):
        """Retrieves training arguments

        Returns:
            Training arguments
        """
        return self._training_args

    def model(self):
        """ Retrieves SKLearn model

        Returns:
            SKLearn model object
        """
        return self._model

    def init_dependencies(self) -> List:
        """Default method where dependencies are returned

         Returns:
             Empty list as default
         """
        return []

    def training_routine(self,
                         history_monitor=None,
                         node_args: Union[dict, None] = None):
        """
        Method training_routine called in Round, to change only if you know what you are doing.

        Args:
        - history_monitor ([type], optional): [description]. Defaults to None.
        - node_args (Union[dict, None]): command line arguments for node. Can include:
            - gpu (bool): propose use a GPU device if any is available. Default False.
            - gpu_num (Union[int, None]): if not None, use the specified GPU device instead of default
              GPU device if this GPU device is available. Default None.
            - gpu_only (bool): force use of a GPU device if any available, even if researcher
              doesnt request for using a GPU. Default False.
                """
        if self._model is None:
            raise FedbiomedTrainingPlanError('model in None')

        # Run preprocesses
        self.__preprocess()

        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning('Node would like to force GPU usage, but sklearn training plan ' +
                           'does not support it. Training on CPU.')

        try:
            self._training_routine_core_loop(self._training_args["epochs"],
                                             history_monitor)
        except FedbiomedTrainingPlanError as e:
            raise e

    def _training_routine_core_loop(self,
                                    epochs: int = 1,
                                    history_monitor: Any = None):
        """
        Training routine core
        Args:
        - model_hook: training_routine_hook of child class {FedSGDClassifier, FedSGDRegressor, FedPerceptron}
        - epochs (integer, optional) : number of training epochs for this round. Defaults to 1
        - history_monitor ([type], optional): [description]. Defaults to None.
        """
        for epoch in range(epochs):
            with _Capturer() as output:
                # Fit model based on model type
                try:
                    self.training_routine_hook()
                except Exception as e:
                    msg = ErrorNumbers.FB605.value + \
                          ": error while fitting the model - " + \
                          str(e)
                    logger.critical(msg)
                    raise FedbiomedTrainingPlanError(msg)
            # Logging training training outputs
            if history_monitor is not None:
                if self._verbose_capture_option:

                    loss = self.evaluate_loss(output, epoch)

                    loss_function = 'Loss ' + self._model.loss if hasattr(self._model, 'loss') else 'Loss'
                    # TODO: This part should be changed after mini-batch implementation is completed
                    history_monitor.add_scalar(metric={loss_function: float(loss)},
                                               iteration=1,
                                               epoch=epoch,
                                               train=True,
                                               num_batches=1,
                                               total_samples=len(self.data),
                                               batch_samples=len(self.data))
                else:
                    # TODO: For clustering; passes inertia value as scalar. It should be implemented when
                    #  KMeans implementation is ready history_monitor.add_scalar('Inertia',
                    #  self._model.inertia_, -1 , epoch) Need to find a way for Bayesian approaches
                    pass

    @staticmethod
    def _evaluate_loss_core(output: StringIO, epoch: int) -> list[float]:
        """
        Evaluate the loss when verbose option _verbose_capture_option is set to True.
        Args:
        - output: output of the scikit-learn models during training
        - epoch: epoch number
        Returns: list[float]: list of loss captured in the output
        """
        _loss_collector = []
        for line in output:
            if len(line.split("loss: ")) == 1:
                continue
            try:
                loss = line.split("loss: ")[-1]
                _loss_collector.append(float(loss))

                # Logging loss values with global logger
                logger.debug('Train Epoch: {} [Batch All Samples]\tLoss: {:.6f}'.format(epoch, float(loss)))
            except ValueError as e:
                logger.error("Value error during monitoring:" + str(e))
            except Exception as e:
                logger.error("Error during monitoring:" + str(e))
        return _loss_collector

    def testing_routine(self,
                        metric: Union[MetricTypes, None],
                        metric_args: Dict[str, Any],
                        history_monitor,
                        before_train: bool):
        """
        Validation routine for SGDSkLearnModel. This method is called by the Round class if validation
        is activated for the Federated training round

        Args:
            metric (MetricType, None): The metric that is going to be used for validation. Should be
                an instance of MetricTypes. If it is None and there is no `testing_step` is defined
                by researcher method will raise an Exception. Defaults to ACCURACY.

            history_monitor (HistoryMonitor): History monitor class of node side to send validation results
                to researcher.

            before_train (bool): If True, this means validation is going to be performed after loading model parameters
              without training. Otherwise, after training.

        """
        # Use accuracy as default metric
        if metric is None:
            metric = MetricTypes.ACCURACY

        if self.testing_data_loader is None:
            msg = ErrorNumbers.FB605.value + ": can not find dataset for validation."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Check validation data loader is exists
        data, target = self.testing_data_loader

        # At the first round model won't have classes_ attribute
        if self._is_classification and not hasattr(self._model, 'classes_'):
            classes = self._classes_from_concatenated_train_test()
            setattr(self._model, 'classes_', classes)

        # Build metrics object
        metric_controller = Metrics()
        tot_samples = len(data)

        # Use validation method defined by user
        if hasattr(self, 'testing_step') and callable(self.testing_step):
            try:
                m_value = self.testing_step(data, target)
            except Exception as err:
                msg = ErrorNumbers.FB605.value + \
                      ": error - " + \
                      str(err)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            # If custom validation step returns None
            if m_value is None:
                msg = ErrorNumbers.FB605.value + \
                      ": metric function has returned None"
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            metric_name = 'Custom'

        # If metric is defined use pre-defined validation for Fed-BioMed
        else:
            if metric is None:
                metric = MetricTypes.ACCURACY
                logger.info(f"No `testing_step` method found in TrainingPlan and `test_metric` is not defined "
                            f"in the training arguments `: using default metric {metric.name}"
                            " for model validation")
            else:
                logger.info(
                    f"No `testing_step` method found in TrainingPlan: using defined metric {metric.name}"
                    " for model validation.")

            try:
                pred = self._model.predict(data)
            except Exception as e:
                msg = ErrorNumbers.FB605.value + \
                      ": error during predicting validation data set - " + \
                      str(e)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            m_value = metric_controller.evaluate(target, pred, metric=metric, **metric_args)
            metric_name = metric.name

        metric_dict = self._create_metric_result_dict(m_value, metric_name=metric_name)

        # For logging in node console
        logger.debug('Validation: [{}/{}] | Metric[{}]: {}'.format(len(target), tot_samples,
                                                                metric.name, m_value))

        # Send scalar values via general/feedback topic
        if history_monitor is not None:
            history_monitor.add_scalar(metric=metric_dict,
                                       iteration=1,  # since there is only one
                                       epoch=None,  # no epoch
                                       test=True,  # means that for sending validation metric
                                       test_on_local_updates=False if before_train else True,
                                       test_on_global_updates=before_train,
                                       total_samples=tot_samples,
                                       batch_samples=len(target),
                                       num_batches=1)

    def _classes_from_concatenated_train_test(self) -> np.ndarray:
        """
        Method for getting all classes from validatino and target dataset. This action is required
        in case of some class only exist in training subset or validation subset

        Returns:
            np.ndarray: numpy array containing unique values from the whole dataset (training + validation dataset)
        """

        target_test = self.testing_data_loader[1] if self.testing_data_loader is not None else np.array([])
        target_train = self.training_data_loader[1] if self.training_data_loader is not None else np.array([])

        target_test_train = np.concatenate((target_test, target_train))

        return np.unique(target_test_train)

    def __preprocess(self) -> None:
        """
        Method for executing registered preprocess that are defined by user.
        """

        for (name, process) in self.pre_processes.items():
            method = process['method']
            process_type = process['process_type']

            if process_type == ProcessTypes.DATA_LOADER:
                self.__process_data_loader(method=method)
            else:
                logger.error(f"Process `{process_type}` is not implemented for the training plan SGBSkLearnModel. "
                             f"Preprocess will be ignored")

    def __process_data_loader(self, method: Callable) -> None:
        """Process handler for data loader kind processes.

        Args:
          method (Callable) : Process method that is going to be executed

        Raises FedbiomedTrainingPlanError:
          - raised when method doesn't have 2 positional arguments
          - Raised if running method fails
          - if dataloader returned by method is not of type: Tuple[np.ndarray, np.ndarray]
          - if dataloaders contained in method output don't contain the same number of samples
       """

        argspec = get_method_spec(method)
        if len(argspec) != 2:
            msg = ErrorNumbers.FB605.value + \
                  ": process for type `PreprocessType.DATA_LOADER`" + \
                  " should have two argument/parameter as inputs/data" + \
                  " and target sets that will be used for training. "
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        try:
            data_loader = method(self.training_data_loader[0], self.training_data_loader[1])
        except Exception as e:
            msg = ErrorNumbers.FB605.value + \
                  ": error while running process method -> " + \
                  method.__name__ + \
                  str(e)
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Debug after running preprocess
        logger.debug(f'The process `{method.__name__}` has been successfully executed.')

        if isinstance(data_loader, tuple) \
                and len(data_loader) == 2 \
                and isinstance(data_loader[0], np.ndarray) \
                and isinstance(data_loader[1], np.ndarray):

            if len(data_loader[0]) == len(data_loader[1]):
                self.training_data_loader = data_loader
                logger.debug(f"Inputs/data and target sets for training routine has been updated by the process "
                             f"`{method.__name__}` ")
            else:
                msg = ErrorNumbers.FB605.value + \
                      ": process error " + \
                      method.__name__ + \
                      " : number of samples of inputs and target sets should be equal "
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

        else:
            msg = ErrorNumbers.FB605.value + \
                  ": process method " + \
                  method.__name__ + \
                  " should return tuple length of two as dataset and" + \
                  " target and both should be and instance of np.ndarray."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    def _compute_support(self, targets: np.ndarray) -> np.ndarray:
        """
        Computes support, i.e. the number of items per
        classes. It is designed from the way scikit learn linear model
        `fit_binary` and `_prepare_fit_binary` have been implemented.

        Args:
            targets (np.ndarray): targets that contain labels
            used for training models

        Returns:
            np.ndarray: support
        """
        support = np.zeros((len(self._model.classes_),))

        # to see how multi classification is done in sklearn, please visit:
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L324   # noqa
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L738   # noqa
        for i, aclass in enumerate(self._model.classes_):
            # in sklearn code, in `fit_binary1`, `i`` seems to be
            # iterated over model.classes_
            # (https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L774)
            # We cannot directly know for each loss that has been logged from scikit learn
            #  which labels it corresponds to. This is our best guest
            idx = targets == aclass
            support[i] = np.sum(targets[targets.astype(int)[idx]])

        return support

    def save(self, filename: str, params: dict = None) -> None:
        """
        Save method for parameter communication, internally is used
        dump and load joblib library methods.

        Args:
            filename: (string) name of the output file
            params: (dictionary) model parameters to save

        Save can be called from Job or Round.
        From round is always called with params.
        From job is called with no params in constructor and with params in update_parameters.

        Torch state_dict has a model_params object. model_params tag is used in the code. This is why this tag is
        used in sklearn case.
        """
        file = open(filename, "wb")
        if params is None:
            dump(self._model, file)
        else:
            if params.get('model_params') is not None:  # called in the Round
                for p in params['model_params'].keys():
                    setattr(self._model, p, params['model_params'][p])
            else:
                for p in params.keys():
                    setattr(self._model, p, params[p])
            dump(self._model, file)
        file.close()

    def load(self, filename: str, to_params: bool = False) -> Dict:
        """Method to load the parameters of a scikit model

        This function updates the `model` attribute with the loaded parameters.
        Load can be called from Job or Round.
        From round is called with no params
        From job is called with  params

        Args:
            filename (string) the name of the file to load
            to_params (boolean) to differentiate a pytorch from a sklearn

        Returns:
            dictionary with the loaded parameters
        """
        di_ret = {}
        file = open(filename, "rb")
        if not to_params:
            self._model = load(file)
            di_ret = self._model
        else:
            self._model = load(file)
            di_ret['model_params'] = {key: getattr(self._model, key) for key in self._param_list}
        file.close()
        return di_ret

    def get_model(self):
        """Get the wrapped scikit-learn model
            Returns:
                the scikit model object (sklearn.base.BaseEstimator)
        """
        return self._model

    def type(self):
        """Getter for training plan type """
        return self.__type

    def after_training_params(self) -> Dict:
        """
        Provide a dictionary with the federated parameters you need to aggregate, refer to
        scikit documentation for a detail of parameters

        Returns:
            the federated parameters (dictionary)
        """
        return {key: getattr(self._model, key) for key in self._param_list}
