import inspect
import os
import sys
import tempfile
import shutil
import atexit
from typing import Union, Callable, List, Dict
import uuid
import re
import time
import copy

import validators

from fedbiomed.common.repository import Repository
from fedbiomed.common.logger import logger
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from fedbiomed.common.torchnn import TorchTrainingPlan
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
                 model: Union[str, Callable] = None,
                 model_path: str = None,
                 training_args: dict = None,
                 model_args: dict = None,
                 data: FederatedDataSet = None):

        """ Constructor of the class.

        Starts a message queue, loads python model file created by researcher
        (through `TrainingPlan`) and saves the loaded model in a temporary file
        (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

        Args:
            reqs (Requests, optional): researcher's requests assigned to nodes.
            Defaults to None.
            nodes (dict, optional): a dict of node_id containing the
            nodes used for training
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
        self._researcher_id = environ['RESEARCHER_ID']
        self._repository_args = {}
        self._training_args = training_args
        self._model_args = model_args
        self._nodes = nodes
        self._training_replies = {}  # will contain all node replies for every round
        self._model_file = None

        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs

        self.last_msg = None
        self._data = data

        # Check dataset quality
        if data is not None:

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

        self.repo = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])
        tmpdirname = tempfile.mkdtemp(prefix=environ['TMP_DIR'])
        atexit.register(lambda: shutil.rmtree(tmpdirname))  # remove `tmpdirname`
        # directory when script will end running (replace
        # `with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as tmpdirname: `)
        self._model_file = tmpdirname + '/my_model_' + str(uuid.uuid4()) + '.py'
        try:
            self.model_instance.save_code(self._model_file)
        except Exception as e:
            logger.error("Cannot save the model to a local tmp dir : " + str(e))
            return

        # upload my_model_xxx.py on HTTP server (contains model definition)
        repo_response = self.repo.upload_file(self._model_file)

        self._repository_args['model_url'] = repo_response['file']

        params_file = tmpdirname + '/my_model_' + str(uuid.uuid4()) + '.pt'
        try:
            self.model_instance.save(params_file)
        except Exception as e:
            logger.error("Cannot save parameters of the model to a local tmp dir : " + str(e))
            return

        # upload my_model_xxx.pt on HTTP server (contains model parameters)
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
        for resp in self._reqs.get_responses(look_for_commands=['model-status'], only_successful = False):
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
        self._params_path = {}
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
            #models_done = self._reqs.get_responses(look_for_commands=['train'])
            models_done = self._reqs.get_responses(look_for_commands=['train', 'error'], only_successful = False)
            #print("=== DEBUG START start_nodes_training_round")
            #print(models_done)
            #print("=== DEBUG STOP  start_nodes_training_round")
            for m in models_done.get_data():  # retrieve all models
                # (there should have as many models done as nodes)

                # manage error messages
                if 'errnum' in m:
                    print("=== DEBUG start_nodes_training_round - ERROR MESSAGE RECEIVED:", m['errnum'])
                    continue

                # only consider replies for our request
                if m['researcher_id'] != environ['RESEARCHER_ID'] or m['job_id'] != self._id or m['node_id'] not in list(self._nodes):
                    continue

                rtime_total = time.perf_counter() - time_start[m['node_id']]

                # TODO : handle error depending on status
                logger.info("Downloading model params after training on " + m['node_id'] + ' - from ' + m['params_url'])
                _, params_path = self.repo.download_file(m['params_url'], 'my_model_' + str(uuid.uuid4()) + '.pt')
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
                self._params_path[r[0]['node_id']] = params_path


    def update_parameters(self, params: dict) -> str:
        """Updates global model parameters after aggregation, by specifying in a
        temporary file (environ['TMP_DIR'] + '/researcher_params_<id>.pt', where <id> is a
        unique and random id)

        Args:
            params (dict): [description]

        Returns:
            str: filename
        """
        try:
            # FIXME: should we specify file extension as a local/global variable ?
            # eg:
            # extension = 'pt'
            # filename = environ['TMP_DIR'] + '/researcher_params_' + str(uuid.uuid4()) + extension

            filename = environ['TMP_DIR'] + '/researcher_params_' + str(uuid.uuid4()) + '.pt'
            self.model_instance.save(filename, params)
            repo_response = self.repo.upload_file(filename)
            self._repository_args['params_url'] = repo_response['file']
        except Exception as e:
            e = sys.exc_info()
            logger.error("Cannot update parameters - Error: " + str(e))
            sys.exit(-1)
        return filename

    def save_state(self, round: int=0):
        """Creates attribute `self.state` containing a
        first state of the job. State will be completed by
        other methods called fro; `Experiment`.

        Args:
            round (int, optional): number of round iteration.
            Defaults to 0.
        """

        self.state = {
            'researcher_id': environ['RESEARCHER_ID'],
            'job_id': self._id,
            'training_data': self._data.data(),
            'training_args': self._training_args,
            'model_args': self._model_args,
            'command': 'train',
            'model_path': self._model_file,
            'params_path': self._params_path,
            'model_class': self._repository_args.get('model_class'),
            'training_replies': self._save_training_replies()
        }

    def _save_training_replies(self) -> Dict[int, List[dict]]:
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
        for node_i, node_entry in enumerate(self._training_replies[last_index]):
            node_id = node_entry.get("node_id")
            converted_training_replies[node_i]['params'] = self._params_path.get(node_id)
        return {int(last_index): converted_training_replies}

    def _load_training_replies(self,
                               training_replies: Dict[int, List[dict]],
                               params_path: Dict[str, str]):
        """Loads training replies from a formatted JSON file,
        so it behaves like a real `training_replies`.
        Gathers parameters values instead of path to paramater files.

        Args:
            training_replies (Dict[int, List[dict]]): JSON formatted
            `training_replies` entry.
            params_path (Dict[str, str]): dictionary of parameter paths (keys)
            mapping node ids (entries).
        """

        # get key
        key = tuple(training_replies.keys())[0]
        if key != int(key):
            # convert string key to integer (converting into JSON
            # change every key type into str type)

            training_replies[int(key)] = training_replies[key]
            #training_replies.pop(key)
            #
            key = int(key)
        loaded_training_replies = {key: Responses([])}
        for node_id, node_i in zip(params_path.keys(),
                                       range(len(training_replies))):
            training_replies[key][node_i]['params'] = self.model_instance.load(params_path[node_id],
                                                                                 to_params=True)

            training_replies[key][node_i]['params_path'] = params_path[node_id]

            loaded_training_replies[key].append(Responses(training_replies[key][node_i]))
        #print(loaded_training_replies)
        self._training_replies = loaded_training_replies

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
