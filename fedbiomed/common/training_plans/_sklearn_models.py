import numpy as np

from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans, ProcessTypes
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger


from ._sklearn_training_plan import SKLearnTrainingPlan

class FedPerceptron(SKLearnTrainingPlan):

    model = Perceptron()

    def __init__(self, model_args: dict = {}):
        """
        Sklearn Perceptron model
        Args:
        - model_args: (dict, optional): model arguments. Defaults to {}
        """
        super().__init__(model_args)

        if 'verbose' not in model_args:
            self.model_args['verbose'] = 1
            self.params.update({'verbose': 1})

        self._is_classification = True

        self._verbose_capture_option = self.model_args['verbose']

        # Instantiate the model
        self.set_init_params()
        self.add_dependency([
                             "from sklearn.linear_model import Perceptron "
                             ])
        print('sklearn models perceptron model get param',self.model.get_params())
        print('perceptron model id ',id(self.model))

    def training_routine_hook(self):
        """
        Training routine of Perceptron
        """
        print('sklearn models enter training routine hook perceptron')
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True
        print('is binary classificaton',self._is_binary_classification)
        print('self.model.get params',self.model.get_params())
        print('self.model',self.model)
        self.model.partial_fit(self.data, self.target, classes=classes)
        print('partial fit ended')

    def set_init_params(self):
        """
        Initialize the model parameter
        """
        self.param_list = ['intercept_','coef_']
        init_params = {
            'intercept_': np.array([0.]) if (self.model_args['n_classes'] == 2) else np.array(
                [0.] * self.model_args['n_classes']),
            'coef_': np.array([0.] * self.model_args['n_features']).reshape(1, self.model_args['n_features']) if (
                    self.model_args['n_classes'] == 2) else np.array(
                [0.] * self.model_args['n_classes'] * self.model_args['n_features']).reshape(self.model_args['n_classes'],
                                                                                   self.model_args['n_features'])
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params:
            setattr(self.model, p, self.params[p])

    def evaluate_loss(self,output,epoch) -> float:
        '''
        Evaluate the loss.
        Args:
        - output: output of the scikit-learn perceptron model during training
        - epoch: epoch number
        Returns: float: the loss captured in the output of its weighted average in case of mutliclass classification
        '''
        print('sklearn models enter evaluate loss perceptron')
        _loss_collector = self._evaluate_loss_core(output, epoch)
        if not self._is_binary_classification:
            support = self._compute_support(self.target)
            loss = np.average(_loss_collector, weights=support)  # perform a weighted average
            logger.warning("Loss plot displayed on Tensorboard may be inaccurate (due to some plain" + \
                           " SGD scikit learn limitations)")
        else:
            loss = _loss_collector[-1]
        return loss


#======

class FedSGDRegressor(SKLearnTrainingPlan):

    model = SGDRegressor()

    def __init__(self, model_args: dict = {}):
        """
        Sklearn SGDRegressor model
        Args:
        - model_args: (dict, optional): model arguments. Defaults to {}
        """
        super().__init__(model_args)

        if 'verbose' not in model_args:
            self.model_args['verbose'] = 1
            self.params.update({'verbose': 1})

        # specific for SGDRegressor
        self._is_regression = True
        self._verbose_capture_option = self.model_args['verbose']

        # Instantiate the model
        self.set_init_params()

        self.add_dependency([
                             "from sklearn.linear_model import SGDRegressor "
                             ])

    def training_routine_hook(self):
        """
        Training routine of SGDRegressor
        """
        (self.data, self.target) = self.training_data_loader
        self.model.partial_fit(self.data, self.target)

    def set_init_params(self):
        """
        Initialize the model parameter
        """
        self.param_list = ['intercept_','coef_']
        init_params = {'intercept_': np.array([0.]),
                       'coef_': np.array([0.] * self.model_args['n_features'])}
        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params:
            setattr(self.model, p, self.params[p])

    def evaluate_loss(self,output,epoch) -> float:
        '''
        Evaluate the loss.
        Args:
        - output: output of the scikit-learn SGDRegressor model during training
        - epoch: epoch number
        Returns: float: the loss captured in the output
        '''

        _loss_collector = self._evaluate_loss_core(output, epoch)
        loss = _loss_collector[-1]
        return loss


#======

class FedSGDClassifier(SKLearnTrainingPlan):

    model = SGDClassifier()

    def __init__(self, model_args : dict = {}):
        """
        Sklearn SGDClassifier model
        Args:
        - model_args: (dict, optional): model arguments. Defaults to {}
        """

        super().__init__(model_args)

        #if verbose is not provided in model_args set it to true and add it to self.params
        if 'verbose' not in model_args:
            self.model_args['verbose'] = 1
            self.params.update({'verbose':1})

        self.is_classification = True
        self._verbose_capture_option = self.model_args['verbose']

        # Instantiate the model
        self.set_init_params()

        self.add_dependency(["from sklearn.linear_model import SGDClassifier "
                             ])

    def training_routine_hook(self):
        """
        Training routine of SGDClassifier
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self):
        """
        Initialize the model parameter
        """
        self.param_list = ['intercept_','coef_']
        init_params = {
            'intercept_': np.array([0.]) if (self.model_args['n_classes'] == 2) else np.array(
                [0.] * self.model_args['n_classes']),
            'coef_': np.array([0.] * self.model_args['n_features']).reshape(1, self.model_args['n_features']) if (
                    self.model_args['n_classes'] == 2) else np.array(
                [0.] * self.model_args['n_classes'] * self.model_args['n_features']).reshape(self.model_args['n_classes'],
                                                                                   self.model_args['n_features'])
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params:
            setattr(self.model, p, self.params[p])

        print('self.params',self.params)
        print('self.model.get_params()', self.model.get_params())
    def evaluate_loss(self,output,epoch) -> float:
        '''
        Evaluate the loss.
        Args:
        - output: output of the scikit-learn SGDClassifier model during training
        - epoch: epoch number
        Returns: float: the loss captured in the output of its weighted average in case of mutliclass classification
        '''
        _loss_collector = self._evaluate_loss_core(output,epoch)
        if not self._is_binary_classification:
            support = self._compute_support(self.target)
            loss = np.average(_loss_collector, weights=support)  # perform a weighted average
            logger.warning("Loss plot displayed on Tensorboard may be inaccurate (due to some plain" + \
                           " SGD scikit learn limitations)")
        else:
            loss = _loss_collector[-1]
        return loss

class FedBernoulliNB(SKLearnTrainingPlan):

    model = BernoulliNB()

    def __init__(self, model_args: dict ={}):
        """
        Sklearn BernoulliNB model
        Args:
        - model_args: (dict, optional): model arguments. Defaults to {}
        """

        super().__init__(model_args)

        self.is_classification = True
        if 'verbose' in model_args:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for BernoulliNB "
                         ": it needs to be implemented")

        self.set_init_params()

        self.add_dependency([
                             "from sklearn.naive_bayes import BernoulliNB"
                             ])

    def training_routine_hook(self):
        """
        Training routine of BernoulliNB
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self):
        """
        Initialize the model parameter
        """
        for p in self.params:
            setattr(self.model, p, self.params[p])
        print('self.params',self.params)
        print('self.model.get_params()', self.model.get_params())

class FedGaussianNB(SKLearnTrainingPlan):

    model = GaussianNB()

    def __init__(self, model_args: dict ={}):
        """
        Sklearn GaussianNB model
        Args:
        - model_args: (dict, optional): model arguments. Defaults to {}
        """
        super().__init__(model_args)
        self.is_classification = True

        if 'verbose' in model_args:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for GaussianNB "
                         ": it needs to be implemeted")

        self.set_init_params()

        self.add_dependency([
                             "from sklearn.naive_bayes  import GaussianNB"
                             ])

    def training_routine_hook(self):
        """
        Training routine of GaussianNB
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(self.data, self.target, classes=classes)


    def set_init_params(self):
        """
        Initialize the model parameter
        """
        for p in self.params:
            setattr(self.model, p, self.params[p])

#############################################################################################3
class FedMultinomialNB(SKLearnTrainingPlan):
    def __init__(self, model_args):

        raise("model not implemented yet")

    def training_routine_hook(self):
        pass

class FedPassiveAggressiveClassifier(SKLearnTrainingPlan):
    def __init__(self, model_args):
        msg = ErrorNumbers.FB605.value + \
              ": model" + __class__.__name__+ " not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass

class FedPassiveAggressiveRegressor(SKLearnTrainingPlan):
    def __init__(self, model_args):
        msg = ErrorNumbers.FB605.value + \
              ": model" + __class__.__name__+ " not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass

class FedMiniBatchKMeans(SKLearnTrainingPlan):
    def __init__(self, model_args):
        msg = ErrorNumbers.FB605.value + \
              ": model" + __class__.__name__+ " not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass

class FedMiniBatchDictionaryLearning(SKLearnTrainingPlan):
    def __init__(self, model_args):
        msg = ErrorNumbers.FB605.value + \
              ": model" + __class__.__name__+ " not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass