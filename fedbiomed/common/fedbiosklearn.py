import inspect
from joblib import dump, load
import torch
import numpy as np
from sklearn.linear_model import SGDRegressor
import json
class SkLearnModel():

    def partial_fit(self,X,y):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def training_data(self, batch_size=48):
        pass

    def after_training_params(self):
        pass

    def training_routine(self, epochs=1, log_interval=10, lr=1e-3, batch_size=50, batch_maxnum=0, dry_run=False,
                         logger=None):
        print('SGD Regressor training batch size ', batch_size)
        (data, target) = self.training_data(batch_size=batch_size)
        print('data is ', data , 'target is ' , target)
        for r in range(epochs):
            self.training_step(data,target)

    def __init__(self,kwargs):
        self.batch_size = 100
        self.dependencies = [   "from fedbiomed.common.fedbiosklearn import SkLearnModel",
                                "import inspect",
                                "import pickle",
                                "import numpy as np",
                                "import pandas as pd",
                                "from sklearn.linear_model import SGDRegressor",
                                "from torchvision import datasets, transforms",

                             ]
        self.dataset_path = None
        self.reg = SGDRegressor(max_iter=kwargs['max_iter'], tol=kwargs['tol'])

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    # provider by fedbiomed
    def save_code(self):

        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)

        # try/except todo
        file = open("my_model.py", "w")
        file.write(content)
        file.close()

    # provided by fedbiomed
    def save(self, filename, params: dict=None):
        if params is None:
            dump(self.reg, open(filename, "wb"))
        else:
            if params.get('model_params') is not None:
                self.reg.coef_ = params['model_params']['coef_']
                self.reg.intercept_ = params['model_params']['intercept_']
            else:
                self.reg.coef_ = params['coef_']
                self.reg.intercept_ = params['intercept_']
            dump(self.reg, open(filename, "wb"))
        if hasattr(self.reg, 'coef_') and hasattr(self.reg, 'intercept_'):
            print('Saving file ', filename, ' with parameters coef ', self.reg.coef_, ' intercept_ ',self.reg.intercept_)
        else:
            print('saving without any params ')



    # provided by fedbiomed
    def load(self, filename, to_params: bool = False):
        print('filename is ', filename)
        di = {}
        if not to_params:
            self.reg = load(open( filename , "rb"))
            if hasattr(self.reg, 'coef_') and hasattr(self.reg, 'intercept_'):
                print('Loaded coef - no parms COEF (ROUND)',  ' with parameters coef ', self.reg.coef_,' intercept_ ',self.reg.intercept_)
            else:
                print('Loaded coef - no parms COEF (ROUND) with no attrs')
            return self.reg
        else:
            self.reg =  load(open(filename, "rb"))
            if hasattr(self.reg, 'coef_') and hasattr(self.reg, 'intercept_'):
                print('Loaded coef - parms COEF (JOB) ', ' with parameters coef ', self.reg.coef_,' intercept_ ',self.reg.intercept_)
            else:
                print('Loaded coef - no parms COEF (ROUND) with no attrs')
            di['model_params'] = {'coef_': self.reg.coef_,'intercept_':self.reg.intercept_}
            return di

    # provided by the fedbiomed / can be overloaded // need WORK
    def logger(self, msg, batch_index, log_interval = 10):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        print('Dataset_path',self.dataset_path)
