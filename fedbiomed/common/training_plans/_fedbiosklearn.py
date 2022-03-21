'''
TrainingPlan definition for sklearn ML framework
'''

from io import StringIO
from joblib import dump, load
import sys
from typing import Union, Tuple, Callable

import numpy as np

from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.metrics.metrics import Metrics

from ._base_training_plan import BaseTrainingPlan

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.constants import ProcessTypes
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

        # 'MultinomialNB': MultinomialNB,
        # 'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
        # 'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
        # 'MiniBatchKMeans': MiniBatchKMeans,
        # 'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
    }

    #
    # SKLEARN method that can return loss value
    #
    _verbose_capture = ['Perceptron',
                        'SGDClassifier',
                        'PassiveAggressiveClassifier',
                        'SGDRegressor',
                        'PassiveAggressiveRegressor',
                        ]

    def __init__(self, model_args: dict = {}):
        """
        Class initializer.

        Args:
        - model_args (dict, optional): model arguments.
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
            logger.info("[TENSORBOARD ERROR]: cannot compute loss for " +
                        model_args['model'] + ": it needs to be implemeted")

        # instanciate the model
        self.m = self.model_map[self.model_type]()

        self.params_sgd = self.m.get_params()
        from_args_sgd_proper_pars = {key: model_args[key] for key in model_args if key in self.params_sgd}
        self.params_sgd.update(from_args_sgd_proper_pars)
        self.param_list = []
        self.set_init_params(model_args)
        self.dataset_path = None
        self._is_classif = False  # whether the model selected is a classifier or not
        self._is_binary_classif = False  # whether the classification is binary or multi classes
        self.__train_data = None
        self.__train_target = None

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
            setattr(self.m, p, init_params[p])

        for p in self.params_sgd:
            setattr(self.m, p, self.params_sgd[p])

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

    def after_training_params(self):
        """
        Provide a dictionary with the federated parameters you need to aggregate, refer to
        scikit documentation for a detail of parameters
        :return the federated parameters (dictionary)
        """
        return {key: getattr(self.m, key) for key in self.param_list}

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
        support = np.zeros((len(self.m.classes_),))

        # to see how multi classification is done in sklearn, please visit:
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L324   # noqa
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L738   # noqa

        for i, aclass in enumerate(self.m.classes_):
            # in sklearn code, in `fit_binary1`, `i`` seems to be
            # iterated over model.classes_
            # (https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L774)
            # We cannot directly know for each loss that has been logged from scikit learn
            #  which labels it corresponds to. This is our best guest
            idx = targets == aclass
            support[i] = np.sum(targets[targets[idx]])

            return support

    def training_routine(self,
                         data_loader: Tuple[np.ndarray, np.ndarray],
                         epochs=1,
                         history_monitor=None,
                         node_args: Union[dict, None] = None):
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
        (self.__train_data, self.__train_target) = data_loader

        # Run preprocesses
        self.__preprocess()

        classes = np.unique(self.__train_target)
        for epoch in range(epochs):
            with _Capturer() as output:
                if self.model_type == 'MultinomialNB' or \
                        self.model_type == 'BernoulliNB' or \
                        self.model_type == 'Perceptron' or \
                        self.model_type == 'SGDClassifier' or \
                        self.model_type == 'PassiveAggressiveClassifier':
                    self.m.partial_fit(self.__train_data, self.__train_target, classes=classes)
                    self._is_classif = True

                elif self.model_type == 'SGDRegressor' or \
                        self.model_type == 'PassiveAggressiveRegressor':  # noqa
                    self.m.partial_fit(self.__train_data, self.__train_target)

                elif self.model_type == 'MiniBatchKMeans' or \
                        self.model_type == 'MiniBatchDictionaryLearning':  # noqa
                    self.m.partial_fit(self.__train_data)

            if history_monitor is not None:
                _loss_collector = []
                if self._is_classif:
                    if classes.shape[0] < 3:
                        # check whether it is a binary classification
                        # or a multiclass classification
                        self._is_binary_classif = True
                if self.model_type in self._verbose_capture:
                    for line in output:
                        # line is of type 'str'
                        if len(line.split("loss: ")) == 1:
                            continue
                        try:
                            loss = line.split("loss: ")[-1]
                            _loss_collector.append(float(loss))
                            # Logging loss values with global logger
                            logger.info('Train Epoch: {} [Batch All Samples]\tLoss: {:.6f}'.format(
                                epoch,
                                float(loss)))

                        except ValueError as e:
                            logger.error("Value error during monitoring:" + str(e))
                        except Exception as e:
                            logger.error("Error during monitoring:" + str(e))

                    if self._is_classif and not self._is_binary_classif:
                        # WARNING: only for plain SGD models in scikit learn
                        # if other models are implemented, should be updated
                        support = self._compute_support(self.__train_target)
                        loss = np.average(_loss_collector, weights=support)  # perform a weighted average

                        logger.warning("Loss plot displayed on Tensorboard may be inaccurate (due to some plain" +
                                       " SGD scikit learn limitations)")

                    # Batch -1 means Batch Gradient Descent, use all samples
                    # TODO: This part should be changed after mini-batch implementation is completed
                    history_monitor.add_scalar('Loss', float(loss), -1, epoch)

                elif self.model_type == "MiniBatchKMeans":
                    # Passes inertia value as scalar. It should be emplemented when KMeans implementation is ready
                    # history_monitor.add_scalar('Inertia', self.m.inertia_, -1 , epoch)
                    pass
                elif self.model_type in ['MultinomialNB', 'BernoulliNB']:
                    # TODO: Need to find a way for Bayesian approaches
                    pass

    def testing_routine(self,
                       data_loader, 
                       metric, 
                       hisory_monitor, 
                       before_train):
        self.__testing_data_loader = data_loader

        # Build metrics object
        metric_controller = Metrics()
        tot_samples = len(self.__testing_data_loader.dataset)
        
        for batch_ndx, (data, target) in enumerate(self.__testing_data_loader):
            
            if hasattr(self, 'testing_step'):
                # use `testing_step` defined by user
                try:
                    eval_res = self.testing_step()
                except Exception as err:
                    raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Error - {str(err)}")
            # use default metric
            elif True:
                # case where score is needed (CLASSIFICATION ONLY)
                
                score = self.m.decision_function(data) 
                # decision_function computes the distance
                # between hyperplan and point. Can be seen as a score
            else:
                # case where label corresponding to proba are needed OR a 
                # regression is performed (CLASSIFICATION AND REGRESSION)
                pred = self.m.predict(data)

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
            dump(self.m, file)
        else:
            if params.get('model_params') is not None:  # called in the Round
                for p in params['model_params'].keys():
                    setattr(self.m, p, params['model_params'][p])
            else:
                for p in params.keys():
                    setattr(self.m, p, params[p])
            dump(self.m, file)
        file.close()

    def load(self, filename, to_params: bool = False):
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
            self.m = load(file)
            di_ret = self.m
        else:
            self.m = load(file)
            di_ret['model_params'] = {key: getattr(self.m, key) for key in self.param_list}
        file.close()
        return di_ret

    def set_dataset_path(self, dataset_path):
        """
          :param dataset_path (string)
        """
        self.dataset_path = dataset_path
        logger.debug('Dataset_path' + str(self.dataset_path))

    def get_model(self):
        """
            :return the scikit model object (sklearn.base.BaseEstimator)
        """
        return self.m

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
                logger.debug(f"Process `{process_type}` is not implemented for the training plan SGBSkLearnModel. "
                             f"Preprocess will be ignored")

    def __process_data_loader(self, method: Callable):

        """
       Process handler for data loader kind processes.

       Args:
           method (Callable) : Process method that is going to be executed

       Raises:
            FedbiomedTrainingPlanError:
       """

        argspec = get_method_spec(method)
        if len(argspec) != 2:
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Process for type "
                                             f"`PreprocessType.DATA_LOADER` should have two argument/parameter as "
                                             f"inputs/data and target sets that will be used for training. ")

        try:
            data_loader = self.preprocess(self.__train_data, self.__train_target)
        except Exception as e:
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB605.value}: Error while running process method -> `{method.__name__}`: "
                f"{str(e)}`")

        # Debug after running preprocess
        logger.debug(f'The process `{method.__name__}` has been successfully executed.')

        if isinstance(data_loader, tuple) \
                and len(data_loader) == 2 \
                and isinstance(data_loader[0], np.ndarray) \
                and isinstance(data_loader[1], np.ndarray):

            if len(data_loader[0]) == len(data_loader[1]):
                self.__train_data = data_loader[0]
                self.__train_target = data_loader[1]
                logger.debug(f"Inputs/data and target sets for training routine has been updated by the process "
                             f"`{method.__name__}` ")
            else:
                raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Process error `{method.__name__}`: "
                                                 f"number of samples of inputs and target sets should be equal ")
        else:
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Process method `{method.__name__}` should "
                                             f"return tuple length of two as dataset and target and both should be and "
                                             f"instance of np.ndarray. ")
