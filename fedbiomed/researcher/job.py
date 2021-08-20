import inspect
import os
import sys
import tempfile
from typing import Union
import uuid
import validators
import re
import time

from fedbiomed.common.repository import Repository
from fedbiomed.researcher.environ import RESEARCHER_ID, TMP_DIR, CACHE_DIR, UPLOADS_URL
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.datasets import FederatedDataSet


class Job:
    """
    This class represents the entity that manage the training part at the clients level
    """    
    def __init__(self,
                reqs: Requests=None, \
                clients: dict=None, \
                model: str = None, \
                model_path: str = None, \
                training_args: dict=None, \
                model_args: dict=None,
                data: FederatedDataSet=None):

        """ Constructor of the class

        Args:
            clients (dict, optional): a dict of client_id containing the clients used for training
            model (string, optional): name of the model class to use for training
            model_path (string, optional) : path to file containing model class code
            training_args (dict, optional): contains training parameters: lr, epochs, batch_size...
                                            Defaults to None.
            model_args (dict, optional): contains output and input feature dimension. 
                                            Defaults to None.
        """        
        self._id = str(uuid.uuid4())
        self._repository_args = {}
        self._training_args = training_args
        self._model_args = model_args
        self._clients = clients
        self._training_replies = {}

        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs


        self.last_msg = None
        self._data = data
        # handle case when model is in a file
        if model_path is not None:
            try:
                model_module = os.path.basename(model_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(model_path))
                exec('from ' + model_module + ' import ' + model)
                sys.path.pop(0)
                model = eval(model)
            except:
                e = sys.exc_info()
                print("Cannot import class ", model, " from path ", model_path, " - Error: ", e)
                sys.exit(-1)

        # create/save model instance
        if inspect.isclass(model):
            if self._model_args is None or len(self._model_args)==0:
                self.model_instance = model()
            else:
                self.model_instance = model(self._model_args)
        else:
            # also handle case where model is an instance of a class
            self.model_instance = model



        self.repo = Repository(UPLOADS_URL, TMP_DIR, CACHE_DIR)
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmpdirname:
            model_file = tmpdirname + '/my_model_' + str(uuid.uuid4()) + '.py'
            try:
                self.model_instance.save_code(model_file)
            except Exception as e:
                print("Cannot save the model to a local tmp dir")
                print(e)
                return

            # upload my_model.py on HTTP server
            repo_response = self.repo.upload_file(model_file)
            self._repository_args['model_url'] = repo_response['file']

            try:
                self.model_instance.save('my_model.pt')
            except Exception as e:
                print("Cannot save parameters of the model to a local tmp dir")
                print(e)
                return

            # upload my_model.pt on HTTP server
            repo_response = self.repo.upload_file('my_model.pt')
            self._repository_args['params_url'] = repo_response['file']

        self._repository_args['model_class'] = re.search("([^\.]*)'>$", str(self.model_instance.__class__)).group(1)

        # Validate fields in each of the arguments
        self.validate_minimal_arguments(self._repository_args, ['model_url', 'model_class', 'params_url'])

    @staticmethod
    def validate_minimal_arguments(obj: dict, fields: Union[tuple, list]):
        """this method validates a given dictionary

        Args:
            obj (dict): object to be validated
            fields (Union[tuple, list]): list of fields that should be present on the obj
        """        
        for f in fields:
            assert f in obj.keys(), f'Field {f} is required in object {obj}. Was not found.'
            if 'url' in f:
                assert validators.url(obj[f]), f'Url not valid: {f}'

    @property
    def model(self):
        return self.model_instance

    @property
    def requests(self):
        return self.reqs

    @property
    def clients(self):
        return self._clients

    @clients.setter
    def clients(self, clients: dict):
        self._clients = clients

    @property
    def training_replies(self):
        return self._training_replies

    @property
    def training_args(self):
        return self._training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self._training_args = training_args

    """ This method should change in sprint8 or as soon as we implement other kind of strategies different than DefaultStrategy"""
    def waiting_for_clients(self, responses: Responses):
        """ this method verify if all clients involved in the job are present Responses

        Args:
            responses (Responses): contains message answers

        Returns:
            bool: True if all clients are present in the Responses object
        """        
        try:
            clients_done = set(responses.dataframe['client_id'])
        except KeyError:
            clients_done = set()

        return not clients_done == set(self._clients)

    def start_clients_training_round(self, round: int):
        """
        this method send training task to clients and waits for the responses
        Args:
            round (int): round of the training
            initial_params (str): url of the init file params
        """    

        headers = {
            'researcher_id': RESEARCHER_ID,
            'job_id': self._id,
            'training_args': self._training_args,
            #'training_data' is set after
            'model_args': self._model_args,
            'command': 'train'
        }

        msg = {**headers, **self._repository_args }
        time_start = {}

        for cli in self._clients:
            msg['training_data'] = { cli: [ ds['dataset_id'] for ds in self._data.data()[cli] ] }
            print('[ RESEARCHER ] Send message to client ', cli, msg)
            time_start[cli] = time.perf_counter()
            self._reqs.send_message( msg , cli)

        # Recollect models trained
        self._training_replies[round] = Responses([])
        while self.waiting_for_clients(self._training_replies[round]):
            models_done = self._reqs.get_responses('train')
            for m in models_done.get_data():
                # only consider replies for our request
                if m['researcher_id'] != RESEARCHER_ID or m['job_id'] != self._id or m['client_id'] not in list(self._clients):
                    continue

                rtime_total = time.perf_counter() - time_start[m['client_id']]

                # TODO : handle error depending on status
                print("Downloading model params after training on ", m['client_id'], '\n\t- from', m['params_url'])
                _, params_path = self.repo.download_file(m['params_url'], 'my_model_' + str(uuid.uuid4()) + '.pt')
                params = self.model_instance.load(params_path, to_params=True)['model_params']
                # TODO: could choose completely different name/structure for job-level data
                timing = m['timing']
                timing['rtime_total'] = rtime_total
                r = Responses({ 'success': m['success'], 'msg': m['msg'], 'dataset_id': m['dataset_id'],
                    'client_id': m['client_id'], 'params_path': params_path, 'params': params,
                    'timing': timing })
                self._training_replies[round].append(r)

    def update_parameters(self, params: dict):
        try:
            filename = TMP_DIR + '/researcher_params_' + str(uuid.uuid4()) + '.pt'
            self.model_instance.save(filename, params)
            repo_response = self.repo.upload_file(filename)
            self._repository_args['params_url'] = repo_response['file']
        except Exception as e:
            e = sys.exc_info()
            print("Cannot update parameters - Error: ", e)
            sys.exit(-1)
        return filename


class localJob:
    """
    This class represents the entity that manage the training part
    """    
    def __init__(self, dataset_path = None, model_class = 'MyTrainingPlan', model_path = None, training_args: dict=None, model_args: dict=None):

        """ Constructor of the class

        Args:
            model_class (string, optional): name of the model class to use for training
            model_path (string, optional) : path to file containing model code
            training_args (dict, optional): contains training parameters: lr, epochs, batch_size...
                                            Defaults to None.
            model_args (dict, optional): contains output and input feature dimension. 
                                            Defaults to None.
        """ 


        self._id = str(uuid.uuid4())
        self._repository_args = {}
        self.__training_args = training_args
        self._model_args = model_args
        self.dataset_path = dataset_path

        # handle case when model is in a file
        if model_path is not None:
            try:
                model_module = os.path.basename(model_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(model_path))
                exec('from ' + model_module + ' import ' + model_class)
                sys.path.pop(0)
                model_class = eval(model_class)
            except:
                e = sys.exc_info()
                print("Cannot import class ", model_class, " from path ", model_path, " - Error: ", e)
                sys.exit(-1)

        # create/save model instance
        if inspect.isclass(model_class):
            if self._model_args is None or len(self._model_args)==0:
                self.model_instance = model_class()
            else:    
                self.model_instance = model_class(self._model_args)
        else:
            self.model_instance = model_class

    @property
    def model(self):
        return self.model_instance

    @property
    def training_args(self):
        return self.__training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self.__training_args = training_args

    def start_training(self):
        """
        this method send training task to clients and waits for the responses
        Args:
            round (int): round of the training
            initial_params (str): url of the init file params
        """    

        for i in self.model_instance.dependencies:
            exec(i, globals()) 
 
        is_failed = False
        error_message = ''

        # Run the training routine
        if not is_failed:
            results = {}
            try:
                
                self.model_instance.set_dataset(self.dataset_path)
                self.model_instance.training_routine(**self.__training_args)
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model: " + str(e)


        if not is_failed:
            try:
                # TODO : should test status code but not yet returned by upload_file
                filename = TMP_DIR + '/local_params_' + str(uuid.uuid4()) + '.pt'
                self.model_instance.save(filename, results)
            except Exception as e:
                is_failed = True
                error_message = "Cannot write results: " + str(e)

        # end : clean the namespace
        try:
            del model
        except:
            pass

        if error_message != '':
            print(error_message)
