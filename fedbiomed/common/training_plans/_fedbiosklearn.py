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


class SGDSkLearnModel(BaseTrainingPlan):
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

    def __init__(self, model_args: dict = {}):
        """
        Class initializer.

        Args:
        - model_args (dict, optional): model arguments. Defaults to {}.
        """
        super().__init__()

        # TODO: Generalize training plan name if there are different training plans for sklearn
        self.__type = TrainingPlans.SkLearnTrainingPlan

        # sklearn.utils.parallel_backend("locky", n_jobs=1, inner_max_num_threads=1)
        self.batch_size = 100  # unused

        self.add_dependency(["import inspect",
                             "import numpy as np",
                             "import pandas as pd",
                             "from fedbiomed.common.training_plans import SGDSkLearnModel",
                             "from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron ",
                             "from sklearn.naive_bayes  import BernoulliNB, GaussianNB",
                             "from fedbiomed.common.data import DataManager",
                             ])

        # default value if passed argument with incorrect type
        if not isinstance(model_args, dict):
            model_args = {}

        if 'model' not in model_args:
            msg = ErrorNumbers.FB303.value + ": SKLEARN model not provided"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        if model_args['model'] not in self.model_map:
            msg = ErrorNumbers.FB303.value + ": SKLEARN model must be one of: " + str(self.model_map)
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        self.model_type = model_args['model']

        # Add verbosity in model_args if not and model is in verbose capturer
        # TODO: check this - verbose doesn't seem to be used ?
        if 'verbose' not in model_args and model_args['model'] in self._verbose_capture:
            model_args['verbose'] = 1

        elif model_args['model'] not in self._verbose_capture:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for " +
                         model_args['model'] + ": it needs to be implemeted")

        # Instantiate the model
        self.model = self.model_map[self.model_type]()
        self.params_sgd = self.model.get_params()
        from_args_sgd_proper_pars = {key: model_args[key] for key in model_args if key in self.params_sgd}
        self.params_sgd.update(from_args_sgd_proper_pars)
        self.param_list = []
        self.set_init_params(model_args)
        self.dataset_path = None

        # Register learning type
        self._is_classification = False
        self._is_binary_classification = False
        self._is_clustering = False
        self._is_regression = False
        if self.model_type in SGDSkLearnModel.CLASSIFICATION_MODELS:
            self._is_classification = True
            self._is_binary_classification = False

        if self.model_type in SGDSkLearnModel.CLUSTERING_MODELS:
            self._is_clustering = True

        if self.model_type in SGDSkLearnModel.REGRESSION_MODELS:
            self._is_regression = True

    def type(self):
        """Getter for training plan type """
        return self.__type

    def set_init_params(self, model_args):
        """
        Initialize model parameters

        Args:
        - model_args (dict) : model parameters
        """
        if self.model_type in ['SGDRegressor']:
            self.param_list = ['intercept_', 'coef_']
            init_params = {'intercept_': np.array([0.]),
                           'coef_': np.array([0.] * model_args['n_features'])}
        elif self.model_type in ['Perceptron', 'SGDClassifier']:
            self.param_list = ['intercept_', 'coef_']
            init_params = {
                'intercept_': np.array([0.]) if (model_args['n_classes'] == 2) else np.array(
                    [0.] * model_args['n_classes']),
                'coef_': np.array([0.] * model_args['n_features']).reshape(1, model_args['n_features']) if (
                    model_args['n_classes'] == 2) else np.array(
                    [0.] * model_args['n_classes'] * model_args['n_features']).reshape(model_args['n_classes'],
                                                                                       model_args['n_features'])
            }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params_sgd:
            setattr(self.model, p, self.params_sgd[p])

    def partial_fit(self, X, y):  # seems unused
        """
        Provide partial fit method of scikit learning model here.

        :param X (ndarray)
        :param y (vector)
        :raise FedbiomedTrainingPlanError if not overloaded
        """
        msg = ErrorNumbers.FB303.value + ": partial_fit must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def after_training_params(self) -> Dict:
        """
        Provide a dictionary with the federated parameters you need to aggregate, refer to
        scikit documentation for a detail of parameters
        :return the federated parameters (dictionary)
        """
        return {key: getattr(self.model, key) for key in self.param_list}

    def _compute_support(self, targets: np.ndarray) -> np.ndarray:
        """
        Computes support, i.e. the number of items per
        classes. It is designed from the way scikit learn linear model
        `fit_binary` and `_prepare_fit_binary` have been implemented.

        Args:
            targets (np.ndarray): targets that contains labels
            used for training models

        Returns:
            np.ndarray: support
        """
        support = np.zeros((len(self.model.classes_),))

        # to see how multi classification is done in sklearn, please visit:
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L324   # noqa
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L738   # noqa

        for i, aclass in enumerate(self.model.classes_):
            # in sklearn code, in `fit_binary1`, `i`` seems to be
            # iterated over model.classes_
            # (https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L774)
            # We cannot directly know for each loss that has been logged from scikit learn
            #  which labels it corresponds to. This is our best guest
            print('aclass',aclass)
            print('i',i)
            idx = targets == aclass
            print('idx',idx)
            print('target[idx]',targets[idx])
            support[i] = np.sum(targets[targets.astype(int)[idx]])

        return support

    def training_routine(self,
                         epochs=1,
                         history_monitor=None,
                         node_args: Union[dict, None] = None):
        # FIXME: remove parameters specific for testing specified in the
        # training routine
        """
        Method training_routine called in Round, to change only if you know what you are doing.

        Args:
        - epochs (integer, optional) : number of training epochs for this round. Defaults to 1
        - history_monitor ([type], optional): [description]. Defaults to None.
        - node_args (Union[dict, None]): command line arguments for node. Can include:
            - gpu (bool): propose use a GPU device if any is available. Default False.
            - gpu_num (Union[int, None]): if not None, use the specified GPU device instead of default
              GPU device if this GPU device is available. Default None.
            - gpu_only (bool): force use of a GPU device if any available, even if researcher
              doesnt request for using a GPU. Default False.
        """
        # issue warning if GPU usage is forced by node : no GPU support for sklearn training
        # plan currently
        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning('Node would like to force GPU usage, but sklearn training plan ' +
                           'does not support it. Training on CPU.')

        #
        # perform sklearn training
        #
        (data, target) = self.training_data_loader

        # Run preprocesses
        self.__preprocess()

        for epoch in range(epochs):
            with _Capturer() as output:
                # Fit model based on model type
                if self._is_classification:
                    classes = self.__classes_from_concatenated_train_test()
                    try:
                        print('self model',self.model)
                        self.model.partial_fit(data, target, classes=classes)
                    except Exception as e:
                        msg = ErrorNumbers.FB605.value + \
                            ": error while fitting the model - " + \
                            str(e)
                        logger.critical(msg)
                        raise FedbiomedTrainingPlanError(msg)
                elif self._is_regression:
                    self.model.partial_fit(data, target)

                elif self._is_clustering:
                    self.model.partial_fit(data)
            print('model get params',self.model.get_params())
            print('output capturer',output)
            # Logging training training outputs
            if history_monitor is not None:
                _loss_collector = []

                # check whether it is a binary classification or a multiclass classification
                if self._is_classification and classes.shape[0] < 3:
                    self._is_binary_classification = True

                if self.model_type in self._verbose_capture:
                    for line in output:
                        if len(line.split("loss: ")) == 1:
                            continue
                        try:
                            loss = line.split("loss: ")[-1]
                            _loss_collector.append(float(loss))

                            # Logging loss values with global logger
                            logger.debug('Train Epoch: {} [Batch All Samples]\tLoss: {:.6f}'.format(epoch,
                                                                                                    float(loss)))
                        except ValueError as e:
                            logger.error("Value error during monitoring:" + str(e))
                        except Exception as e:
                            logger.error("Error during monitoring:" + str(e))

                    if self._is_classification and not self._is_binary_classification:
                        # WARNING: only for plain SGD models in scikit learn
                        # if other models are implemented, should be updated
                        print('target', target)
                        support = self._compute_support(target)
                        loss = np.average(_loss_collector, weights=support)  # perform a weighted average

                        logger.warning("Loss plot displayed on Tensorboard may be inaccurate (due to some plain" +
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

    def save(self, filename, params: dict = None):
        """
        Save method for parameter communication, internally is used
        dump and load joblib library methods.
        :param filename (string)
        :param params (dictionary) model parameters to save

        Save can be called from Job or Round.
            From round is always called with params.
            From job is called with no params in constructor and
            with params in update_parameters.

            Torch state_dict has a model_params object. model_params tag
            is used in the code. This is why this tag is
            used in sklearn case.
        """
        file = open(filename, "wb")
        if params is None:
            dump(self.model, file)
        else:
            if params.get('model_params') is not None:  # called in the Round
                for p in params['model_params'].keys():
                    setattr(self.model, p, params['model_params'][p])
            else:
                for p in params.keys():
                    setattr(self.model, p, params[p])
            dump(self.model, file)
        file.close()

    def load(self, filename, to_params: bool = False) -> Dict:
        """
        Method to load the updated parameters of a scikit model
        Load can be called from Job or Round.
        From round is called with no params
        From job is called with  params
        :param filename (string)
        :param to_params (boolean) to differentiate a pytorch from a sklearn
        :return dictionary with the loaded parameters.
        """
        di_ret = {}
        file = open(filename, "rb")
        if not to_params:
            self.model = load(file)
            di_ret = self.model
        else:
            self.model = load(file)
            di_ret['model_params'] = {key: getattr(self.model, key) for key in self.param_list}
        file.close()
        return di_ret

    def get_model(self):
        """
            :return the scikit model object (sklearn.base.BaseEstimator)
        """
        return self.model

    def __preprocess(self):
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

    def __process_data_loader(self, method: Callable):

        """
        Process handler for data loader kind processes.

        Args:
          method (Callable) : Process method that is going to be executed

        Raises FedbiomedTrainingPlanError:
          - raised when method doesnot have 2 positional arguments
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
