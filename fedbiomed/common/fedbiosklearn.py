import inspect
from joblib import dump, load
import numpy as np
from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

class SGDSkLearnModel():

    def set_init_params(self, kwargs):
        """
            Initialize model parameters
            :param kwargs (dictionary) containing model parameters
        """
        if self.model_type in ['SGDRegressor']:
            self.param_list = ['intercept_','coef_']
            init_params = {'intercept_': np.array([0.]), 
                           'coef_':  np.array([0.]*kwargs['n_features'])}
        elif self.model_type in ['Perceptron', 'SGDClassifier']:
            self.param_list = ['intercept_','coef_']
            init_params = {'intercept_': np.array([0.]) if (kwargs['n_classes'] == 2) else np.array([0.]*kwargs['n_classes']),
                           'coef_':  np.array([0.]*kwargs['n_features']).reshape(1,kwargs['n_features']) if (kwargs['n_classes'] == 2) else np.array([0.]*kwargs['n_classes']*kwargs['n_features']).reshape(kwargs['n_classes'],kwargs['n_features'])  }

        for p in self.param_list:
            setattr(self.m, p, init_params[p])
        
        for p in self.params_sgd:
            setattr(self.m, p, self.params_sgd[p])

    def partial_fit(self,X,y):
        """
            Provide partial fit method of scikit learning model here.
            :param X (ndarray)
            :param y (vector)
            :raise NotImplementedError if developer do not implement this method.
        """
        raise NotImplementedError('Partial fit must be implemented')

    def training_data(self):
        """
            Perform in this method all data reading and data transformations you need.
            At the end you should provide a couple (X,y) as indicated in the partial_fit
            method of the scikit learn class.
            :raise NotImplementedError if developer do not implement this method.
        """
        raise NotImplementedError('Training data must be implemented')

    def after_training_params(self):
        """Provide a dictionary with the federated parameters you need to aggregate, refer to
            scikit documentation for a detail of parameters
            :return the federated parameters (dictionary)
        """
        return {key: getattr(self.m, key) for key in self.param_list}

    def training_routine(self, epochs=1,logger=None):
        """
            Method training_routine called in Round, to change only if you know what you are doing.
            :param epochs (integer)
        """
        (data, target) = self.training_data()
        for _ in range(epochs):
            if self.model_type == 'MultinomialNB' or self.model_type == 'BernoulliNB' or self.model_type == 'Perceptron' or self.model_type == 'SGDClassifier' or self.model_type == 'PassiveAggressiveClassifier' :
                self.m.partial_fit(data,target, classes = np.unique(target))
            elif self.model_type == 'SGDRegressor' or self.model_type == 'PassiveAggressiveRegressor':
                self.m.partial_fit(data,target)
            elif self.model_type == 'MiniBatchKMeans' or self.model_type == 'MiniBatchDictionaryLearning':
                self.m.partial_fit(data)

    def __init__(self,kwargs):
        """
           Class initializer.
           :param kwargs (dictionary) containing model parameters
        """
        self.batch_size = 100
        self.model_map = {'MultinomialNB', 'BernoulliNB', 'Perceptron', 'SGDClassifier', 'PassiveAggressiveClassifier',
                          'SGDRegressor', 'PassiveAggressiveRegressor', 'MiniBatchKMeans',
                          'MiniBatchDictionaryLearning'}

        self.dependencies = [   "from fedbiomed.common.fedbiosklearn import SGDSkLearnModel",
                                "import inspect",
                                "import numpy as np",
                                "import pandas as pd",
                             ]
        if kwargs['model'] not in self.model_map:
            print('model must be one of, ', self.model_map)
        else:
            self.model_type = kwargs['model']
            self.m = eval(self.model_type)()
            self.params_sgd = self.m.get_params()
            from_kwargs_sgd_proper_pars = {key: kwargs[key] for key in kwargs if key in self.params_sgd}
            self.params_sgd.update(from_kwargs_sgd_proper_pars)
            self.param_list = []
            self.set_init_params(kwargs)
            self.dataset_path = None


    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        """
           Add new dependency to this class.
           :param dep (string) dependency to add.
        """
        self.dependencies.extend(dep)
        pass

    '''Save the code to send to nodes '''
    def save_code(self, filename):
        """Save the class code for this training plan to a file
           :param filename (string): path to the destination file
        """
        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)

        # try/except todo
        file = open(filename, "w")
        file.write(content)
        file.close()

    def save(self, filename, params: dict=None):
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
            if params.get('model_params') is not None: # called in the Round
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
        file = open( filename , "rb")
        if not to_params:
            self.m = load(file)
            di_ret =  self.m
        else:
            self.m =  load(file)
            di_ret['model_params'] = {key: getattr(self.m, key) for key in self.param_list}
        file.close()
        return di_ret

    def set_dataset(self, dataset_path):
        """
          :param dataset_path (string)
        """
        self.dataset_path = dataset_path
        print('Dataset_path',self.dataset_path)

    def get_model(self):
        """
            :return the scikit model object (sklearn.base.BaseEstimator)
        """
        return self.m