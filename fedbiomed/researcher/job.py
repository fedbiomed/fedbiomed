import inspect
import os
import sys
import tempfile
import shutil
import atexit
from typing import Union, Callable
import uuid
import re
import time
import copy

import validators

from fedbiomed.common.repository import Repository
from fedbiomed.common.logger import logger
from fedbiomed.researcher.environ import RESEARCHER_ID, TMP_DIR, CACHE_DIR, UPLOADS_URL
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.datasets import FederatedDataSet


class Job:
    """
    This class represents the entity that manage the training part at
    the nodes level
    """
    def __init__(self,
                 reqs: Requests = None,
                 clients: dict = None,
                 model: Union[str, Callable] = None,
                 model_path: str = None,
                 training_args: dict = None,
                 model_args: dict = None,
                 data: FederatedDataSet = None, 
                 save_breakpoint:bool=True):

        """ Constructor of the class.

        Starts a message queue, loads python model file created by researcher
        (through `TrainingPlan`) and saves the loaded model in a temporary file
        (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

        Args:
            reqs (Requests, optional): researcher's requests assigned to nodes.
            Defaults to None.
            clients (dict, optional): a dict of client_id containing the
            clients used for training
            model (Union[str, Callable], optional): name of the model class
            to use for training
            model_path (string, optional) : path to file containing model
            class code
            training_args (dict, optional): contains training parameters:
            lr, epochs, batch_size...Defaults to None.
            model_args (dict, optional): contains output and input feature
                                        dimension.Defaults to None.
            data (FederatedDataset, optional): . Defaults to None.

        """
        self._id = str(uuid.uuid4())  # creating a unique job id
        self._repository_args = {}
        self._training_args = training_args
        self._model_args = model_args
        self._clients = clients
        self._training_replies = {}  # will contain all node replies for every round
        self._model_file = None
        
        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs

        self.last_msg = None
        self._data = data
        # handle case when model is in a file
        if model_path is not None:
            try:
                # import model from python file
                model_module = os.path.basename(model_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(model_path))
                exec('from ' + model_module + ' import ' + model)
                sys.path.pop(0)
                model = eval(model)
            except Exception:
                e = sys.exc_info()
                logger.critical("Cannot import class " + model + " from path " +  model_path, " - Error: " + str(e))
                sys.exit(-1)

        # create/save model instance (ie TrainingPlan)
        if inspect.isclass(model):
            # case if `model` is a class
            if self._model_args is None or len(self._model_args) == 0:
                self.model_instance = model()  # contains TrainingPlan
            else:
                self.model_instance = model(self._model_args)
        else:
            # also handle case where model is an instance of a class
            self.model_instance = model

        self.repo = Repository(UPLOADS_URL, TMP_DIR, CACHE_DIR)
        tmpdirname = tempfile.mkdtemp(prefix=TMP_DIR)
        atexit.register(lambda: shutil.rmtree(tmpdirname))  # remove `tmpdirname` 
        # directory when script will end running
        #with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmpdirname:
        self._model_file = tmpdirname + '/my_model_' + str(uuid.uuid4()) + '.py'
        print("tmpdirname", tmpdirname)
        try:
            self.model_instance.save_code(self._model_file)
        except Exception as e:
            logger.error("Cannot save the model to a local tmp dir : " + str(e))
            return

        # upload my_model_xxx.py on HTTP server
        repo_response = self.repo.upload_file(self._model_file)
        
        self._repository_args['model_url'] = repo_response['file']

        params_file = tmpdirname + '/my_model_' + str(uuid.uuid4()) + '.pt'
        try:
            self.model_instance.save(params_file)
        except Exception as e:
            logger.error("Cannot save parameters of the model to a local tmp dir : " + str(e))
            return

        # upload my_model_xxx.pt on HTTP server
        repo_response = self.repo.upload_file(params_file)
        self._repository_args['params_url'] = repo_response['file']

        # (below) regex: matches a character not present among "^", "\", "."
        # characters at the end of string.
        self._repository_args['model_class'] = re.search("([^\.]*)'>$", str(self.model_instance.__class__)).group(1)
        
        # Validate fields in each argument
        self.validate_minimal_arguments(self._repository_args,
                                        ['model_url', 'model_class', 'params_url'])
        # FIXME: (above) the constructor of a class usually shouldnt call one of the method class in its definition

    @staticmethod
    def validate_minimal_arguments(obj: dict, fields: Union[tuple, list]):
        """this method validates a given dictionary

        Args:
            obj (dict): object to be validated
            fields (Union[tuple, list]): list of fields that should be present
            on the obj
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

    """ This method should change in sprint8 or as soon as we implement other
    kind of strategies different than DefaultStrategy"""
    def waiting_for_clients(self, responses: Responses) -> bool:
        """ this method verifies if all clients involved in the job are
        present and Responding

        Args:
            responses (Responses): contains message answers

        Returns:
            bool: False if all clients are present in the Responses object.
            True if waiting for at least one client.
        """
        try:
            clients_done = set(responses.dataframe['client_id'])
        except KeyError:
            clients_done = set()

        return not clients_done == set(self._clients)

    def start_clients_training_round(self, round: int):
        """
        this method sends training task to clients and waits for the responses
        Args:
            round (int): current number of round the algorithm is performing
            (a round is considered to be all the
            training steps of a federated model between 2 aggregations).

        """
        self._params_path = {}
        headers = {
            'researcher_id': RESEARCHER_ID,
            'job_id': self._id,
            'training_args': self._training_args,
            #'training_data' is set after
            'model_args': self._model_args,
            'command': 'train'
        }

        msg = {**headers, **self._repository_args}
        
        time_start = {}
        
        for cli in self._clients:
            msg['training_data'] = { cli: [ ds['dataset_id'] for ds in self._data.data()[cli] ] }
            logger.info('Send message to client ' + str(cli) + " - " + str(msg))
            time_start[cli] = time.perf_counter()
            self._reqs.send_message(msg, cli)  # send request to node

        # Recollect models trained
        self._training_replies[round] = Responses([])
        while self.waiting_for_clients(self._training_replies[round]):
            # collect nodes responses from researcher request 'train'
            # (wait for all clients with a ` while true` loop)
            models_done = self._reqs.get_responses('train')
            for m in models_done.get_data():  # retrieve all models
                # (there should have as many models done as nodes)

                # only consider replies for our request
                if m['researcher_id'] != RESEARCHER_ID or m['job_id'] != self._id or m['client_id'] not in list(self._clients):
                    continue

                rtime_total = time.perf_counter() - time_start[m['client_id']]

                # TODO : handle error depending on status
                logger.info("Downloading model params after training on " + m['client_id'] + ' - from ' + m['params_url'])
                _, params_path = self.repo.download_file(m['params_url'], 'my_model_' + str(uuid.uuid4()) + '.pt')
                params = self.model_instance.load(params_path, to_params=True)['model_params']
                # TODO: could choose completely different name/structure for
                # job-level data
                timing = m['timing']
                timing['rtime_total'] = rtime_total
                r = Responses({'success': m['success'],
                               'msg': m['msg'],
                               'dataset_id': m['dataset_id'],
                               'client_id': m['client_id'],
                               'params_path': params_path,
                               'params': params,
                               'timing': timing})
                self._training_replies[round].append(r)  # add new replies
            
                self._params_path[r[0]['client_id']] = params_path
    

    def update_parameters(self, params: dict) -> str:
        """Updates global model parameters after aggregation, by specifying in a
        temporary file (TMP_DIR + '/researcher_params_<id>.pt', where <id> is a
        unique and random id)

        Args:
            params (dict): [description]

        Returns:
            str: [description]
        """
        try:
            # FIXME: should we specify file extension as a local/global variable ?
            # eg:
            # extension = 'pt'
            # filename = TMP_DIR + '/researcher_params_' + str(uuid.uuid4()) + extension

            filename = TMP_DIR + '/researcher_params_' + str(uuid.uuid4()) + '.pt'
            self.model_instance.save(filename, params)
            repo_response = self.repo.upload_file(filename)
            self._repository_args['params_url'] = repo_response['file']
        except Exception as e:
            e = sys.exc_info()
            logger.error("Cannot update parameters - Error: " + str(e))
            sys.exit(-1)
        return filename

    def save_state(self, round: int=0):
        
        training_data = {last_reply["dataset_id"]: last_reply["client_id"] for \
                         last_reply in self._training_replies[round]}
        
        self.state = {
            'researcher_id': RESEARCHER_ID,
            'job_id': self._id,
            'training_data': training_data,
            'training_args': self._training_args,
            'model_args': self._model_args,
            'command': 'train',
            'model_path': self._model_file,
            'params_path': self._params_path,
            #'model_class': type(self.model_instance).__name__,
            'model_class': self._repository_args.get('model_class'),
            'training_replies': self._save_training_replies()
        }
        
    def _save_training_replies(self) -> list:
        """saves last values training replies variable, and replace
        pytroch tensor / numpy arrays by path files pointing to
        tensor files (these tensor files contain pytorch tensor / numpy arrays)

        Returns:
            list: `_training_replies` variable containing path files towards
            pytorch / numpy arrays instead of Tensors/Arrays values (so it can 
            be saved with JSON).
        """
        last_index = max(self._training_replies.keys())
        converted_training_replies = copy.deepcopy(
                                    self._training_replies[last_index].data
                                    )
        # training_replies saving facility
        for client_i, client_entry in enumerate(self._training_replies[last_index]):
            client_id = client_entry.get("client_id")
            params = self._params_path.get(client_id)
            converted_training_replies[client_i]['params'] = self._params_path.get(client_id)
        return converted_training_replies
         
class localJob:
    """
    This class represents the entity that manage the training part.
    LocalJob is the version of Job but applied locally on a local dataset (thus not involving any network).
    It is only used to compare results to a Federated approach, using networks.
    """
    def __init__(self, dataset_path = None,
                 model_class: str='MyTrainingPlan',
                 model_path: str=None,
                 training_args: dict=None,
                 model_args: dict=None):

        """ Constructor of the class

        Args:
            dataset_path (): . Defaults to None.
            model_class (string, optional): name of the model class to use for training. Defaults to
            'MyTrainingPlan'.
            model_path (string, optional) : path to file containing model code. Defaults to None.
            training_args (dict, optional): contains training parameters: lr, epochs, batch_size...
                                            Defaults to None.
            model_args (dict, optional): contains output and input feature dimension.
                                            Defaults to None.
        """


        self._id = str(uuid.uuid4())
        self._repository_args = {}
        self._localjob_training_args = training_args
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
                logger.critical("Cannot import class " + model_class, " from path ", model_path, " - Error: " + str(e))
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
        return self._localjob_training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self._localjob_training_args = training_args

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
                self.model_instance.training_routine(**self._localjob_training_args)
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model in job : " + str(e)

        if not is_failed:
            try:
                # TODO : should test status code but not yet returned
                # by upload_file
                filename = TMP_DIR + '/local_params_' + str(uuid.uuid4()) + '.pt'
                self.model_instance.save(filename, results)
            except Exception as e:
                is_failed = True
                error_message = "Cannot write results: " + str(e)

        # end : clean the namespace
        try:
            del model
        except Exception:
            pass

        if error_message != '':
            logger.error(error_message)
