import inspect
from joblib import dump, load
from fedbiomed.common.logger import logger

class PythonModelPlan():
    def __init__(self,
                 kwargs):
        """
           Class initializer.
           Here the researcher should define the custom model by initializing a dictionary (self.params_dict),
           where keys are the expected model parameters do be optimized with the training_routine method.
           :kwargs (dictionary) containing model arguments required for your custom model
        """
        # list dependencies of the model
        self.dependencies = [
                             "from fedbiomed.common.pythonmodel import PythonModelPlan"
                             ]

        self.dataset_path = None
        self.params_dict = {}
        pass
        
    #################################################
    def training_routine(self, 
                         n_iterations: int=None,
                         logger=None):

        """ 
        Method detailing the training routine in each node. It has to be defined for each custom python model.
        Args:
            n_iterations (int): the number of iterations or epochs for the current round. Defaults to None
            :raise NotImplementedError if developer do not implement this method.
        """
        raise NotImplementedError('Training data must be implemented')

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
            used here too.
        """
        file = open(filename, "wb")
        if params is None:
            dump(self.params_dict, file)
        else:
            if params.get('model_params') is not None: # called in the Round
                self.update_params(params['model_params'])
                dump(self.params_dict, file)
            else:
                self.update_params(params)
                dump(self.params_dict, file)
        file.close()

    def load(self, filename, to_params: bool = False):
        """
        Method to load the updated parameters.
        load can be called from Job or Round.
        From round is called with no params
        From job is called with params
           :param filename (string)
           :param to_params (boolean)
           :return dictionary with the loaded parameters.
        """
        di_ret = {}
        file = open( filename , "rb")
        if not to_params:
            params_dict = load(file)
            self.update_params(params_dict)
            di_ret =  params_dict
        else:
            params_dict =  load(file)
            self.update_params(params_dict)
            di_ret['model_params'] = params_dict
        file.close()
        return di_ret

    def set_dataset(self,
                    dataset_path: str,
                    multi_view: str = None):
        """
           :param dataset_path (string)
        """
        self.dataset_path = dataset_path
        logger.debug('Dataset_path' + str(self.dataset_path))
        self.is_multi_view = True if multi_view == "multi_view" else False
        logger.debug(f'is Dataset  multi view ? {self.is_multi_view}')

    def get_model(self):
        """
           :return the model parameters
        """
        all_params = self.params_dict
        return all_params

    def training_data(self):
        """
            Perform in this method all data reading and data transformations you need.
            At the end you should provide a tuple (X_obs,Xk,ViewsX,y), where: 
            X_obs is the training dataset, 
            Xk is a list containing the k-specific dataframe if the k-view has been observed or 'NaN' otherwise,
            ViewsX is the indicator function for observed vies (ViewsX[k]=1 if view k is observed, 0 otherwise)
            y the corresponding labels.
            Note: The dataset is normalized using min max scaler if model_args['norm'] is true
            Note: labels are not used for optimization purposes, but can be useful for performance evaluation.
            :raise NotImplementedError if developer do not implement this method.
        """
        raise NotImplementedError('Training data must be implemented')

    def update_params(self,params_dict):
        """
           Update model parameters
           :param params_dict (dict)
        """
        self.params_dict.update(params_dict)

    def after_training_params(self):
        """Provide a dictionary with the federated parameters you need to aggregate
           :return the federated parameters (dictionary)
        """
        return self.params_dict
