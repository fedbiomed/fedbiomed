import inspect
import os
import sys
import tempfile
import shutil
import atexit
from typing import Union, Callable, List, Dict, Type
import uuid
import re
import time
import copy

import validators

from fedbiomed.common.repository import Repository
from fedbiomed.common.logger import logger
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from fedbiomed.common.torchnn import TorchTrainingPlan
from fedbiomed.researcher.filetools import  create_unique_link, \
            create_unique_file_link
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.common.message import ResearcherMessages

class Job:
    """
    This class represents the entity that manage the training part at
    the nodes level
    """
    def __init__(self,
                 reqs: Requests = None,
                 nodes: dict = None,
                 model: Union[Type[Callable], Callable] = None,
                 model_path: str = None,
                 training_args: dict = None,
                 model_args: dict = None,
                 data: FederatedDataSet = None,
                 keep_files_dir: str = None):

        """ Constructor of the class.

        Starts a message queue, loads python model file created by researcher
        (through `TrainingPlan`) and saves the loaded model in a temporary file
        (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

        Args:
            reqs (Requests, optional): researcher's requests assigned to nodes.
            Defaults to None.
            nodes (dict, optional): a dict of node_id containing the
            nodes used for training
            model (Union[Type[Callable], Callable], optional): name of the model class
            or object (instance of the model class) to use for training.
            model_path (string, optional) : path to file containing model
            class code
            training_args (dict, optional): contains training parameters:
            lr, epochs, batch_size...Defaults to None.
            model_args (dict, optional): contains output and input feature
                                        dimension.Defaults to None.
            data (FederatedDataset, optional): . Defaults to None.
            keep_files_dir(str, optional): directory for storing files created by the job
                that we want to keep beyond the execution of the job.
                Defaults to None, files are not kept after the end of the job.

        """
        self._id = str(uuid.uuid4())  # creating a unique job id
        self._researcher_id = environ['RESEARCHER_ID']
        self._repository_args = {}
        self._training_args = training_args
        self._model_args = model_args
        self._nodes = nodes
        self._training_replies = {}  # will contain all node replies for every round
        self._model_file = None # path to local file containing model code
        self._model_params_file = None # path to local file containing current version of aggregated params

        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ['TMP_DIR'])
            atexit.register(lambda: shutil.rmtree(self._keep_files_dir)) # remove directory
                # when script ends running (replace
                # `with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as self._keep_files_dir: `)

        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs

        self.last_msg = None
        self._data = data

        # Check dataset quality
        if self._data is not None:

            self.check_data_quality()


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
                logger.critical("Cannot import class " + model + " from path " +  model_path + " - Error: " + str(e))
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

        self.repo = Repository(environ['UPLOADS_URL'], self._keep_files_dir, environ['CACHE_DIR'])
        
        self._model_file = self._keep_files_dir + '/my_model_' + str(uuid.uuid4()) + '.py'
        try:
            self.model_instance.save_code(self._model_file)
        except Exception as e:
            logger.error("Cannot save the model to a local tmp dir : " + str(e))
            return
        # upload my_model_xxx.py on HTTP server (contains model definition)
        repo_response = self.repo.upload_file(self._model_file)
        self._repository_args['model_url'] = repo_response['file']

        self._model_params_file = self._keep_files_dir + '/aggregated_params_init_' + str(uuid.uuid4()) + '.pt'
        try:
            self.model_instance.save(self._model_params_file)
        except Exception as e:
            logger.error("Cannot save parameters of the model to a local tmp dir : " + str(e))
            return
        # upload aggregated_params_init_xxx.pt on HTTP server (contains model parameters)
        repo_response = self.repo.upload_file(self._model_params_file)
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
        return self._reqs

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: dict):
        self._nodes = nodes

    @property
    def training_replies(self):
        return self._training_replies

    @property
    def training_args(self):
        return self._training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self._training_args = training_args

    @property
    def model_file(self):
        return self._model_file 
    


    # TODO: After refactoring experiment this method can be created 
    # directly in the Experiment class. Since it requires 
    # node ids and model_url to send model approve status it is created
    # in job class  
    def check_model_is_approved_by_nodes(self):

        """ Method for checking whether model is approved or not.  This method send 
            `model-status` request to the nodes. It should be run before running experiment. 
            So, researchers can find out if their model has been approved
        """

        message = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'model_url' : self._repository_args['model_url'],
            'command': 'model-status'
        }  

        responses = []
        replied_nodes = []
        node_ids = self._data.node_ids

        # Send message to each node that has been found after dataset search reqeust
        for cli in node_ids:
            logger.info('Sending request to node ' + \
                                    str(cli) + " to check model is approved or not")
            self._reqs.send_message(
                        ResearcherMessages.request_create(message).get_dict(), 
                        cli) 

        # Wait for responses
        for resp in self._reqs.get_responses(look_for_command='model-status', only_successful = False):
            responses.append(resp)
            replied_nodes.append(resp.get('node_id'))

            if resp.get('success') == True: 
                if resp.get('approval_obligation') == True:
                    if resp.get('is_approved') == True:
                        logger.info(f'Model has been approved by the node: {resp.get("node_id")}')
                    else:
                        logger.warning(f'Model has NOT been approved by the node: {resp.get("node_id")}')
                else:
                    logger.info(f'Model approval is not required by the node: {resp.get("node_id")}')
            else: 
                logger.warning(f"Node : {resp.get('node_id')} : {resp.get('msg')}")

        # Get the nodes that haven't replied model-status request
        non_replied_nodes = list(set(node_ids) - set(replied_nodes))
        if non_replied_nodes:
            logger.warning(f"Request for checking model status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while runing your experiment. ")

        return responses


    """ This method should change in sprint8 or as soon as we implement other
    kind of strategies different than DefaultStrategy"""
    def waiting_for_nodes(self, responses: Responses) -> bool:
        """ this method verifies if all nodes involved in the job are
        present and Responding

        Args:
            responses (Responses): contains message answers

        Returns:
            bool: False if all nodes are present in the Responses object.
            True if waiting for at least one node.
        """
        try:
            nodes_done = set(responses.dataframe['node_id'])
        except KeyError:
            nodes_done = set()

        return not nodes_done == set(self._nodes)

    def start_nodes_training_round(self, round: int):
        """
        this method sends training task to nodes and waits for the responses
        Args:
            round (int): current number of round the algorithm is performing
            (a round is considered to be all the
            training steps of a federated model between 2 aggregations).

        """
        headers = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'training_args': self._training_args,
            #'training_data' is set after
            'model_args': self._model_args,
            'command': 'train'
        }

        msg = {**headers, **self._repository_args}

        time_start = {}

        for cli in self._nodes:
            msg['training_data'] = { cli: [ ds['dataset_id'] for ds in self._data.data()[cli] ] }
            logger.info('Send message to node ' + str(cli) + " - " + str(msg))
            time_start[cli] = time.perf_counter()
            self._reqs.send_message(msg, cli)  # send request to node

        # Recollect models trained
        self._training_replies[round] = Responses([])
        while self.waiting_for_nodes(self._training_replies[round]):
            # collect nodes responses from researcher request 'train'
            # (wait for all nodes with a ` while true` loop)
            models_done = self._reqs.get_responses('train')
            for m in models_done.get_data():  # retrieve all models
                # (there should have as many models done as nodes)

                # only consider replies for our request
                if m['researcher_id'] != environ['RESEARCHER_ID'] or m['job_id'] != self._id or m['node_id'] not in list(self._nodes):
                    continue

                rtime_total = time.perf_counter() - time_start[m['node_id']]

                # TODO : handle error depending on status
                logger.info("Downloading model params after training on " + m['node_id'] + ' - from ' + m['params_url'])
                _, params_path = self.repo.download_file(m['params_url'], 'node_params_' + str(uuid.uuid4()) + '.pt')
                params = self.model_instance.load(params_path, to_params=True)['model_params']
                # TODO: could choose completely different name/structure for
                # job-level data
                timing = m['timing']
                timing['rtime_total'] = rtime_total
                r = Responses({'success': m['success'],
                               'msg': m['msg'],
                               'dataset_id': m['dataset_id'],
                               'node_id': m['node_id'],
                               'params_path': params_path,
                               'params': params,
                               'timing': timing})
                self._training_replies[round].append(r)  # add new replies


    def update_parameters(self, params: dict={}, filename: str=None) -> str:
        """Updates global model aggregated parameters in `params`, by saving them
        to a file `filename` (unless it already exists), then upload file to the repository
        so that params are ready to be sent to the nodes for the next training round.
        If a `filename` is given (file exists) it has precedence over `params`.

        Args:
            params (dict, optional): data structure containing the
                new version of the aggregated parameters for this job,
            filename (str, optional) : path to the file containing the
                new version of the aggregated parameters for this job,

        Returns:
            str: filename
        """
        try:
            if not filename:
                if not params:
                    raise ValueError('Bad arguments for update_parameters, filename or params is needed')
                filename = self._keep_files_dir + '/aggregated_params_' + str(uuid.uuid4()) + '.pt'
                self.model_instance.save(filename, params)
            
            repo_response = self.repo.upload_file(filename)
            self._repository_args['params_url'] = repo_response['file']
            self._model_params_file = filename
        except Exception as e:
            e = sys.exc_info()
            logger.error("Cannot update parameters - Error: " + str(e))
            sys.exit(-1)
        return self._model_params_file

    def save_state(self, breakpoint_path: str, round: int=0):
        """Creates current state of the job to be included in a breakpoint.
        Includes creating links to files included in the job state.

        Args:
            breakpoint_path (str): path to the existing breakpoint directory
            round (int, optional): number of round iteration.
            Defaults to 0.

        Returns:
            dict: job current state information for breakpoint
        """

        state = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'training_data': self._data.data(),
            'training_args': self._training_args,
            'model_args': self._model_args,
            'model_path': self._model_file,
            'model_class': self._repository_args.get('model_class'),
            'model_params_path': self._model_params_file,
            'training_replies': self._save_training_replies(self._training_replies)
        }

        state['model_params_path'] = create_unique_link(
            breakpoint_path,
            'aggregated_params_current', '.pt',
            os.path.join('..', os.path.basename(state["model_params_path"]))
            )

        for round_replies in state['training_replies']:
            for response in round_replies:
                node_params_path = create_unique_file_link(breakpoint_path,
                                            response['params_path'])
                response['params_path'] = node_params_path


        return state

    def load_state(self, saved_state: dict=None):
        """Load breakpoint status for a Job from a saved state

        Args:
            saved_state (dict): breakpoint content
        """
        self._id = saved_state.get('job_id')
        self.update_parameters(filename=saved_state.get('model_params_path'))
        self._training_replies = self._load_training_replies(
                    saved_state.get('training_replies'),
                    self.model_instance.load
                    )
        self._researcher_id = saved_state.get('researcher_id')


    @staticmethod
    def _save_training_replies(training_replies: Dict[int, Responses]) \
                -> List[List[dict]]:
        """Extracts a copy of `training_replies` and
        prepares it for saving in breakpoint
        - strip unwanted fields
        - structure as list/dict so it can be saved with JSON

        Args:
            - training_replies (Dict[int, Responses]) : training replies of
              already executed rounds of the job

        Returns:
            List[List[dict]] : extract from `training_replies` formatted for breakpoint
        """
        converted_training_replies = []
        
        for round in training_replies.keys():
            training_reply = copy.deepcopy(training_replies[round].data)
            # we want to strip some fields for the breakpoint
            for node in training_reply:
                del node['params']
            converted_training_replies.append(training_reply)

        return converted_training_replies

    @staticmethod
    def _load_training_replies(
                bkpt_training_replies: List[List[dict]],
                func_load_params: Callable
                ) -> Dict[int, Responses]:
        """Read training replies from a formatted breakpoint file,
        and build a job training replies data structure .

        Args:
            - training_replies (List[List[dict]]): extract from
              training replies saved in breakpoint
            - func_load_params (Callable) : function for loading parameters
              from file to training replies data structure

        Returns: 
            Dict[int, Responses] : training replies of already executed rounds of the job
        """

        training_replies = {}
        for round in range(len(bkpt_training_replies)):
            loaded_training_reply = Responses(bkpt_training_replies[round])
            # reload parameters from file params_path
            for node in loaded_training_reply:
                node['params'] = func_load_params(
                    node['params_path'], to_params=True)['model_params']

            training_replies[round] = loaded_training_reply

        return training_replies

    def check_data_quality(self):

        """Compare datasets that has been found in different nodes.
        """
        data = self._data.data()
        # If there are more than two nodes ready for the job
        if len(data.keys()) > 1:

            # Frist check data types are same based on searched tags
            logger.info('Checking data quality of federated datasets...')

            data_types = [] # CSV, Image or default
            shapes = [] # dimensions
            dtypes = [] # variable types for CSV datasets

            # Extract features into arrays for comparison
            for data_list in data.items():
                for feature in data_list[1]:
                    data_types.append(feature["data_type"])
                    dtypes.append(feature["dtypes"])
                    shapes.append(feature["shape"])

            assert len(set(data_types)) == 1,\
                 f'Diferent type of datasets has been loaded with same tag: {data_types}'

            if data_types[0] == 'csv':
                assert len(set([s[1] for s in shapes])) == 1, \
                        f'Number of columns of federated datasets do not match {shapes}.'

                dtypes_t = list(map(list, zip(*dtypes)))
                for t in dtypes_t:
                    assert len(set(t)) == 1, \
                         f'Variable data types do not match in federated datasets {dtypes}'

            elif data_types[0] == 'images':

                shapes_t = list(map(list, zip(*[s[2:] for s in shapes])))
                dim_state = True
                for s in shapes_t:
                    if len(set(s)) != 1:
                        dim_state = False

                if not dim_state:
                    logger.error(f'Dimensions of the images in federated datasets \
                                 do not match. Please consider using resize. {shapes} ')


                if len(set([ k[1] for k in shapes])) != 1:
                    logger.error(f'Color channels of the images in federated \
                                    datasets do not match. {shapes}')

            # If it is default MNIST dataset pass
            else:
                pass

        pass

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
                logger.critical("Cannot import class " + model_class + " from path " + model_path + " - Error: " + str(e))
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
        this method send training task to nodes and waits for the responses
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
                filename = environ['TMP_DIR'] + '/local_params_' + str(uuid.uuid4()) + '.pt'
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
