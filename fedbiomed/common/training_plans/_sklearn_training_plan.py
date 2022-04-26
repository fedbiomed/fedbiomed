'''
TrainingPlan definition for sklearn ML framework
'''

import sys
import numpy as np

from typing import Any, Dict, Union, Callable
from io import StringIO
from joblib import dump, load

from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from ._base_training_plan import BaseTrainingPlan

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans, ProcessTypes
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.utils import get_method_spec


class _Capturer(list):
    """
    Capturing class for output of the scikit-learn models during training
    when the verbose is set to true.
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
    #
    # mapping between model name and model class
    model_map = {
        "SGDRegressor": SGDRegressor,
        "SGDClassifier": SGDClassifier,
        "Perceptron": Perceptron,
        "BernoulliNB": BernoulliNB,
        "GaussianNB": GaussianNB,

        # Not implemented
        # 'MultinomialNB': MultinomialNB,
        # 'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
        # 'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
        # 'MiniBatchKMeans': MiniBatchKMeans,
        # 'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
    }

    # Learning Algorithms
    CLUSTERING_MODELS = ('MiniBatchKMeans', 'MiniBatchDictionaryLearning')
    CLASSIFICATION_MODELS = ('MultinomialNB', 'BernoulliNB', 'Perceptron', 'SGDClassifier',
                             'PassiveAggressiveClassifier')
    REGRESSION_MODELS = ('SGDRegressor', 'PassiveAggressiveRegressor')

    # SKLEARN method that can return loss value
    _verbose_capture = ['Perceptron',
                        'SGDClassifier',
                        'PassiveAggressiveClassifier',
                        'SGDRegressor',
                        'PassiveAggressiveRegressor']

    def __init__(self, sklearn_model,training_routine_hook, model_args: dict = {}, verbose_possibility: bool = False):
        """
        Class initializer.

        Args:
        - model_args (dict, optional): model arguments. Defaults to {}.
        """
        super().__init__()


        if not isinstance(model_args, dict):
            msg = ErrorNumbers.FB303.value + ": SKLEARN model_args is not a dict"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        self._model_args = model_args
        self._training_routine_hook = training_routine_hook
        try:
            self.model = sklearn_model()
        except:
            msg = str(sklearn_model) + ": is not a SKLEARN model"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        self.params_sgd = self.model.get_params()
        self.params_sgd.update({key: model_args[key] for key in model_args if key in self.params_sgd})
        self.param_list = []
        self.dataset_path = None

        self._is_classification = False
        self._is_regression = False
        self._is_clustering = False
        self._is_binary_classification = False

        if 'verbose' not in model_args and verbose_possibility:
            model_args['verbose'] = 1

        elif not verbose_possibility:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for " +
                         self.model + ": it needs to be implemented")

        # Instantiate the model
        self.set_init_params(model_args)

    def training_routine(self,epochs=1,
                             history_monitor=None,
                             node_args: Union[dict, None] = None):

        # Run preprocesses
        self.__preprocess()

        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning('Node would like to force GPU usage, but sklearn training plan ' +
                           'does not support it. Training on CPU.')

        try:
            self._training_routine_core_loop(self._training_routine_hook,
                                             epochs,
                                             history_monitor)
        except FedbiomedTrainingPlanError as e:
            raise e

    def _training_routine_core_loop(self,
                                    model_hook,
                                    epochs=1,
                                    history_monitor=None):

        for epoch in range(epochs):
            with _Capturer() as output:
                # Fit model based on model type
                try:
                    model_hook()
                except Exception as e:
                    msg = ErrorNumbers.FB605.value + \
                          ": error while fitting the model - " + \
                          str(e)
                    logger.critical(msg)
                    raise FedbiomedTrainingPlanError(msg)

            # Logging training training outputs
            if history_monitor is not None:
                _loss_collector = []

            if self._verbose_capture_option:
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

            if self._is_classification and not self._is_binary_classification:
                # WARNING: only for plain SGD models in scikit learn
                # if other models are implemented, should be updated
                support = self._compute_support(target)
                loss = np.average(_loss_collector, weights=support)  # perform a weighted average

                logger.warning("Loss plot displayed on Tensorboard may be inaccurate (due to some plain" + \
                               " SGD scikit learn limitations)")

                loss_function = 'Loss ' + self.model.loss if hasattr(self.model, 'loss') else 'Loss'
                # TODO: This part should be changed after mini-batch implementation is completed
                history_monitor.add_scalar(metric={loss_function: float(loss)},
                                           iteration=1,
                                           epoch=epoch,
                                           train=True,
                                           num_batches=1,
                                           total_samples=len(data),
                                           batch_samples=len(data))
            else:
                # TODO: For clustering; passes inertia value as scalar. It should be implemented when
                #  KMeans implementation is ready history_monitor.add_scalar('Inertia',
                #  self.model.inertia_, -1 , epoch) Need to find a way for Bayesian approaches
                pass



    def testing_routine(self,
                        metric: Union[MetricTypes, None],
                        metric_args: Dict[str, Any],
                        history_monitor,
                        before_train: bool):
        """
        Testing routine for SGDSkLearnModel. This method is called by the Round class if testing
        is activated for the Federated training round

        Args:
            metric (MetricType, None): The metric that is going to be used for evaluation. Should be
                an instance of MetricTypes. If it is None and there is no `testing_step` is defined
                by researcher method will raise an Exception. Defaults to ACCURACY.

            history_monitor (HistoryMonitor): History monitor class of node side to send evaluation results
                to researcher.

            before_train (bool): If True, this means testing is going to be performed after loading model parameters
              without training. Otherwise, after training.

        """
        # Use accuracy as default metric
        if metric is None:
            metric = MetricTypes.ACCURACY

        if self.testing_data_loader is None:
            msg = ErrorNumbers.FB605.value + ": can not find dataset for testing."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Check testing data loader is exists
        data, target = self.testing_data_loader

        # At the first round model won't have classes_ attribute
        if self._is_classification and not hasattr(self.model, 'classes_'):
            classes = self.__classes_from_concatenated_train_test()
            setattr(self.model, 'classes_', classes)

        # Build metrics object
        metric_controller = Metrics()
        tot_samples = len(data)

        # Use testing method defined by user
        if hasattr(self, 'testing_step') and callable(self.testing_step):
            try:
                m_value = self.testing_step(data, target)
            except Exception as err:
                msg = ErrorNumbers.FB605.value + \
                    ": error - " + \
                    str(err)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            # If custom evaluation step returns None
            if m_value is None:
                msg = ErrorNumbers.FB605.value + \
                    ": metric function has returned None"
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            metric_name = 'Custom'

        # If metric is defined use pre-defined evaluation for Fed-BioMed
        else:
            if metric is None:
                metric = MetricTypes.ACCURACY
                logger.info(f"No `testing_step` method found in TrainingPlan and `test_metric` is not defined "
                            f"in the training arguments `: using default metric {metric.name}"
                            " for model evaluation")
            else:
                logger.info(
                    f"No `testing_step` method found in TrainingPlan: using defined metric {metric.name}"
                    " for model evaluation.")

            try:
                pred = self.model.predict(data)
            except Exception as e:
                msg = ErrorNumbers.FB605.value + \
                    ": error during predicting test data set - " + \
                    str(e)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            m_value = metric_controller.evaluate(target, pred, metric=metric, **metric_args)
            metric_name = metric.name

        metric_dict = self._create_metric_result_dict(m_value, metric_name=metric_name)

        # For logging in node console
        logger.debug('Testing: [{}/{}] | Metric[{}]: {}'.format(len(target), tot_samples,
                                                                metric.name, m_value))

        # Send scalar values via general/feedback topic
        if history_monitor is not None:
            history_monitor.add_scalar(metric=metric_dict,
                                       iteration=1,  # since there is only one
                                       epoch=None,  # no epoch
                                       test=True,  # means that for sending test metric
                                       test_on_local_updates=False if before_train else True,
                                       test_on_global_updates=before_train,
                                       total_samples=tot_samples,
                                       batch_samples=len(target),
                                       num_batches=1)

    def __classes_from_concatenated_train_test(self) -> np.ndarray:
        """
        Method for getting all classes from test and target dataset. This action is required
        in case of some class only exist in training subset or testing subset

        Returns:
            np.ndarray: numpy array containing unique values from the whole dataset (training + testing dataset)
        """

        target_test = self.testing_data_loader[1] if self.testing_data_loader is not None else np.array([])
        target_train = self.training_data_loader[1] if self.training_data_loader is not None else np.array([])

        target_test_train = np.concatenate((target_test, target_train))

        return np.unique(target_test_train)

