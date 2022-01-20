import sys
from io import StringIO
import inspect
from typing import Union

from joblib import dump, load
import numpy as np
from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.logger import logger

class _Capturer(list):

    """ Capturing class for output of the scikitlearn models during training
    when the verbose is set to true.
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # Remove it from memory
        sys.stdout = self._stdout

class SGDSkLearnModel():

    def set_init_params(self, model_args):
        """
            Initialize model parameters

            Args:
            - model_args (dict) : model parameters
        """
        if self.model_type in ['SGDRegressor']:
            self.param_list = ['intercept_','coef_']
            init_params = {'intercept_': np.array([0.]),
                           'coef_':  np.array([0.]*model_args['n_features'])}
        elif self.model_type in ['Perceptron', 'SGDClassifier']:
            self.param_list = ['intercept_','coef_']
            init_params = {
                'intercept_': np.array([0.]) if (model_args['n_classes'] == 2) else np.array([0.]*model_args['n_classes']),
                'coef_':  np.array([0.]*model_args['n_features']).reshape(1,model_args['n_features']) if (model_args['n_classes'] == 2) else np.array([0.]*model_args['n_classes']*model_args['n_features']).reshape(model_args['n_classes'],model_args['n_features'])
            }

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

    def training_routine(self,
                         epochs=1,
                         monitor=None,
                         node_args: Union[dict, None] = None):
        """
            Method training_routine called in Round, to change only if you know what you are doing.

            Args:
            - epochs (integer, optional) : number of training epochs for this round. Defaults to 1
            - monitor ([type], optional): [description]. Defaults to None.
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
            logger.warning('Node would like to force GPU usage, but sklearn training plan '
                + 'does not support it. Training on CPU.')

        #
        # perform sklearn training
        #
        (data, target) = self.training_data()
        for epoch in range(epochs):
            with _Capturer() as output:
                if self.model_type == 'MultinomialNB' or self.model_type == 'BernoulliNB' or self.model_type == 'Perceptron' or self.model_type == 'SGDClassifier' or self.model_type == 'PassiveAggressiveClassifier' :
                    self.m.partial_fit(data,target, classes = np.unique(target))
                elif self.model_type == 'SGDRegressor' or self.model_type == 'PassiveAggressiveRegressor':
                    self.m.partial_fit(data,target)
                elif self.model_type == 'MiniBatchKMeans' or self.model_type == 'MiniBatchDictionaryLearning':
                    self.m.partial_fit(data)
            
            if monitor is not None:
                if self.model_type in self._verbose_capture:
                    for line in output:
                        if(len(line.split("loss: ")) == 1):
                            continue
                        try:
                            loss = line.split("loss: ")[-1]
                            
                            # Logging loss values with global logger 
                            logger.info('Train Epoch: {} [Batch All Samples]\tLoss: {:.6f}'.format(
                                            epoch,
                                            float(loss)))
                            
                            # Batch -1 means Batch Gradient Descent, use all samples
                            # This part should be changed after mini-batch implementation is completed
                            monitor.add_scalar('Loss', float(loss), -1 , epoch)

                        except ValueError as e:
                            logger.error("Value error:" + e)
                        except Exception as e:
                            logger.error("Error during monitoring:" + e)
                elif self.model_type == "MiniBatchKMeans":
                    # Passes inertia value as scalar. It should be emplemented when KMeans implementation is ready 
                    # monitor.add_scalar('Inertia', self.m.inertia_, -1 , epoch)
                    pass
                elif self.model_type in ['MultinomialNB','BernoulliNB']:
                    # Need to find a way for Bayesian approaches 
                    pass


    def __init__(self, model_args: dict = {}):
        """
        Class initializer.

        Args:
        - model_args (dict, optional): model arguments.
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
        
        # default value if passed argument with incorrect type
        if not isinstance(model_args, dict):
            model_args = {}

        if 'model' not in model_args or model_args['model'] not in self.model_map:
            logger.error('model must be one of, ' +  str(self.model_map))
        else:
            self.model_type = model_args['model']

            # Sklearn mothods that returns loss value when the verbose flag is provided
            self._verbose_capture = ['Perceptron', 'SGDClassifier', 
                                     'PassiveAggressiveClassifier', 
                                     'SGDRegressor', 
                                     'PassiveAggressiveRegressor']

            # Add verbosity in model_args if not and model is in verbose capturer
            # TODO: check this - verbose doesn't seem to be used ?
            if 'verbose' not in model_args and model_args['model'] in self._verbose_capture:
                model_args['verbose'] = 1
            
            elif model_args['model'] not in self._verbose_capture:
                logger.info("[TENSORBOARD ERROR]: cannot compute loss for " +\
                    model_args['model'] + ": it needs to be implemeted")
            self.m = eval(self.model_type)()
            self.params_sgd = self.m.get_params()
            from_args_sgd_proper_pars = {key: model_args[key] for key in model_args if key in self.params_sgd}
            self.params_sgd.update(from_args_sgd_proper_pars)
            self.param_list = []
            self.set_init_params(model_args)
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
        logger.debug('Dataset_path' + str(self.dataset_path))

    def get_model(self):
        """
            :return the scikit model object (sklearn.base.BaseEstimator)
        """
        return self.m
