# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import atexit
import copy
import inspect
import os
import re
import sys
import shutil
import tempfile
import time
import uuid
import importlib
import random
from typing import Any, Tuple, Union, Callable, List, Dict, Type

import validators

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.exceptions import FedbiomedRepositoryError, FedbiomedDataQualityCheckError
from fedbiomed.common.logger import logger
from fedbiomed.common.repository import Repository
from fedbiomed.common.training_args import TrainingArgs

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_unique_link, create_unique_file_link
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses


class Job:
    """
    Represents the entity that manage the training part at  the nodes level

    Starts a message queue, loads python model file created by researcher (through
    [`training_plans`][fedbiomed.common.training_plans]) and saves the loaded model in a temporary file
    (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

    """

    def __init__(self,
                 reqs: Requests = None,
                 nodes: dict = None,
                 training_plan_class: Union[Type[Callable], str] = None,
                 training_plan_path: str = None,
                 training_args: TrainingArgs = None,
                 model_args: dict = None,
                 data: FederatedDataSet = None,
                 keep_files_dir: str = None):

        """ Constructor of the class

        Args:
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            nodes: A dict of node_id containing the nodes used for training
            training_plan_class: instance or class of the TrainingPlan.
            training_plan_path: Path to file containing model class code
            training_args: Contains training parameters; lr, epochs, batch_size.
            model_args: Contains output and input feature dimension
            data: Federated datasets
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        Raises:
            NameError: If model is not defined or if the class can not to be inspected
        """

        self._id = str(uuid.uuid4())  # creating a unique job id
        self._researcher_id = environ['RESEARCHER_ID']
        self._repository_args = {}
        self._training_args = training_args
        self._model_args = model_args
        self._nodes = nodes
        self._training_replies = {}  # will contain all node replies for every round
        self._model_file = None  # path to local file containing model code
        self._model_params_file = None  # path to local file containing current version of aggregated params
        self._training_plan_class = training_plan_class
        self._training_plan = None
        self._aggregator_args = None

        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ['TMP_DIR'])
            atexit.register(lambda: shutil.rmtree(self._keep_files_dir))  # remove directory
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

        # Model is mandatory
        if self._training_plan_class is None:
            mess = "Missing training plan class name or instance in Job arguments"
            logger.critical(mess)
            raise NameError(mess)

        # handle case when model is in a file
        if training_plan_path is not None:
            try:
                # import model from python file

                model_module = os.path.basename(training_plan_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(training_plan_path))

                module = importlib.import_module(model_module)
                tr_class = getattr(module, self._training_plan_class)
                self._training_plan_class = tr_class
                sys.path.pop(0)

            except Exception as e:
                e = sys.exc_info()
                logger.critical(f"Cannot import class {self._training_plan_class} from "
                                f"path {training_plan_path} - Error: {str(e)}")
                sys.exit(-1)

        # check class is defined
        try:
            _ = inspect.isclass(self._training_plan_class)
        except NameError:
            mess = f"Cannot find training plan for Job, training plan class {self._training_plan_class} is not defined"
            logger.critical(mess)
            raise NameError(mess)

        # create/save TrainingPlan instance
        if inspect.isclass(self._training_plan_class):
            self._training_plan = self._training_plan_class()  # contains TrainingPlan

        else:
            self._training_plan = self._training_plan_class
        self._training_plan.post_init(model_args={} if self._model_args is None else self._model_args,
                                      training_args=self._training_args)

        # find the name of the class in any case
        # (it is `model` only in the case where `model` is not an instance)
        self._training_plan_name = self._training_plan.__class__.__name__

        self.repo = Repository(environ['UPLOADS_URL'], self._keep_files_dir, environ['CACHE_DIR'])

        self._training_plan_file = os.path.join(self._keep_files_dir, 'my_model_' + str(uuid.uuid4()) + '.py')
        try:
            self._training_plan.save_code(self._training_plan_file)
        except Exception as e:
            logger.error("Cannot save the training plan to a local tmp dir : " + str(e))
            return

        # upload my_model_xxx.py on repository server (contains model definition)
        repo_response = self.repo.upload_file(self._training_plan_file)

        self._repository_args['training_plan_url'] = repo_response['file']

        self._model_params_file = self._keep_files_dir + '/aggregated_params_init_' + str(uuid.uuid4()) + '.pt'
        try:
            self._training_plan.save(self._model_params_file)
        except Exception as e:
            logger.error("Cannot save parameters of the model to a local tmp dir : " + str(e))
            return
        # upload aggregated_params_init_xxx.pt on repository server (contains model parameters)
        repo_response = self.repo.upload_file(self._model_params_file)
        self._repository_args['params_url'] = repo_response['file']

        # (below) regex: matches a character not present among "^", "\", "."
        # characters at the end of string.
        self._repository_args['training_plan_class'] = self._training_plan_name

        # Validate fields in each argument
        self.validate_minimal_arguments(self._repository_args,
                                        ['training_plan_url', 'training_plan_class', 'params_url'])
        # FIXME: (above) the constructor of a class usually shouldnt call one of the method class in its definition

    @staticmethod
    def validate_minimal_arguments(obj: dict, fields: Union[tuple, list]):
        """ Validates a given dictionary by given mandatory fields.

        Args:
            obj: Object to be validated
            fields: List of fields that should be present on the obj
        """
        for f in fields:
            assert f in obj.keys(), f'Field {f} is required in object {obj}. Was not found.'
            if 'url' in f:
                assert validators.url(obj[f]), f'Url not valid: {f}'

    @property
    def id(self):
        return self._id

    @property
    def training_plan_name(self):
        return self._training_plan_name

    @property
    def training_plan(self):
        return self._training_plan

    @property
    def training_plan_file(self):
        return self._training_plan_file

    @property
    def requests(self):
        return self._reqs

    @property
    def aggregator_args(self):
        return self._aggregator_args

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
        return self._training_args.dict()

    @training_args.setter
    def training_args(self, training_args: TrainingArgs):
        self._training_args = training_args

    def check_training_plan_is_approved_by_nodes(self) -> List:

        """ Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        """

        message = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'training_plan_url': self._repository_args['training_plan_url'],
            'command': 'training-plan-status'
        }

        responses = Responses([])
        replied_nodes = []
        node_ids = self._data.node_ids()

        # Send message to each node that has been found after dataset search request
        for cli in node_ids:
            logger.info('Sending request to node ' +
                        str(cli) + " to check model is approved or not")
            self._reqs.send_message(
                message,
                cli)

        # Wait for responses
        for resp in self._reqs.get_responses(look_for_commands=['training-plan-status'], only_successful=False):
            responses.append(resp)
            replied_nodes.append(resp.get('node_id'))

            if resp.get('success') is True:
                if resp.get('approval_obligation') is True:
                    if resp.get('status') == TrainingPlanApprovalStatus.APPROVED.value:
                        logger.info(f'Training plan has been approved by the node: {resp.get("node_id")}')
                    else:
                        logger.warning(f'Training plan has NOT been approved by the node: {resp.get("node_id")}.' +
                                       f'Training plan status : {resp.get("status")}')
                else:
                    logger.info(f'Training plan approval is not required by the node: {resp.get("node_id")}')
            else:
                logger.warning(f"Node : {resp.get('node_id')} : {resp.get('msg')}")

        # Get the nodes that haven't replied training-plan-status request
        non_replied_nodes = list(set(node_ids) - set(replied_nodes))
        if non_replied_nodes:
            logger.warning(f"Request for checking training plan status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while running your experiment. ")

        return responses

    # TODO: This method should change in the future or as soon as we implement other of strategies different
    #   than DefaultStrategy

    def waiting_for_nodes(self, responses: Responses) -> bool:
        """ Verifies if all nodes involved in the job are present and Responding

        Args:
            responses: contains message answers

        Returns:
            False if all nodes are present in the Responses object. True if waiting for at least one node.
        """
        try:
            nodes_done = set(responses.dataframe()['node_id'])
        except KeyError:
            nodes_done = set()

        return not nodes_done == set(self._nodes)

    def upload_aggregator_args(self,
                               args_thr_msg: Union[Dict[str, Dict[str, Any]], dict],
                               args_thr_files: Union[Dict[str, Dict[str, Any]], dict]) -> Dict[str, Dict[str, Any]]:
        """Uploads aggregator metadata to the Repository and updates the mqtt message accordingly.

        Args:
            args_thr_msg: dictionary containing metadata about the aggregation
                strategy, useful to transfer some data when it's required by am aggregator. First key should be the
                node_id, and sub-dictionary should be parameters to be sent through MQTT messaging system. This
                dictionary may be modified by this function with additional metadata about other metadata
                transferred via the Repository.
            args_thr_files: dictionary containing metadata about aggregation strategy, to be transferred
                via the Repository's HTTP API, as opposed to the mqtt system. Format is the same as
                aggregator_args_thr_msg .

        Returns:
            The updated dictionary with metadata to be introduced in the mqtt message.
        """
        for node_id, aggr_params in args_thr_files.items():
            for arg_name, aggr_param in aggr_params.items():
                if arg_name == 'aggregator_name':
                    continue
                args_thr_msg[node_id][arg_name] = {}
                args_thr_msg[node_id][arg_name]['arg_name'] = arg_name  # name of the argument to look at

                filename, url = self.update_parameters(aggr_param, None,
                                                       is_model_params=False,
                                                       variable_name=arg_name)
                args_thr_msg[node_id][arg_name]['filename'] = filename  # path to the file with the parameters
                args_thr_msg[node_id][arg_name]['url'] = url

        return args_thr_msg

    def start_nodes_training_round(self,
                                   round_: int,
                                   aggregator_args_thr_msg: Dict[str, Dict[str, Any]],
                                   aggregator_args_thr_files: Dict[str, Dict[str, Any]],
                                   secagg_arguments: Union[Dict, None] = None,
                                   do_training: bool = True):
        """ Sends training request to nodes and waits for the responses

        Args:
            round_: current number of round the algorithm is performing (a round is considered to be all the
                training steps of a federated model between 2 aggregations).
            aggregator_args_thr_msg: dictionary containing some metadata about the aggregation
                strategy, useful to transfer some data when it's required by am aggregator. First key should be the
                node_id, and sub-dictionary sould be parameters to be sent through MQTT messaging system
            aggregator_args_thr_files: dictionary containing metadata about aggregation strategy, to be transferred
                via the Repository's HTTP API, as opposed to the mqtt system. Format is the same as
                aggregator_args_thr_msg .
            secagg_arguments: Secure aggregation ServerKey context id
            do_training: if False, skip training in this round (do only validation). Defaults to True.
        """

        # Assign empty dict to secagg arguments if it is None
        if secagg_arguments is None:
            secagg_arguments = {}

        headers = {'researcher_id': self._researcher_id,
                   'job_id': self._id,
                   'training_args': self._training_args.dict(),
                   'training': do_training,
                   'model_args': self._model_args,
                   'round': round_,
                   'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
                   'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
                   'secagg_random': secagg_arguments.get('secagg_random'),
                   'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
                   'command': 'train',
                   'aggregator_args': {}}

        msg = {**headers, **self._repository_args}
        time_start = {}

        # pass heavy aggregator params through file exchange system
        self.upload_aggregator_args(aggregator_args_thr_msg, aggregator_args_thr_files)

        for cli in self._nodes:
            msg['dataset_id'] = self._data.data()[cli]['dataset_id']

            if aggregator_args_thr_msg:
                # add aggregator parameters to message header
                msg['aggregator_args'] = aggregator_args_thr_msg[cli]

            if not do_training:
                logger.info(f'\033[1mSending request\033[0m \n'
                            f'\t\t\t\t\t\033[1m To\033[0m: {str(cli)} \n'
                            f'\t\t\t\t\t\033[1m Request: \033[0m:Perform final validation on '
                            f'aggregated parameters \n {5 * "-------------"}')
            else:
                msg_print = {key: value for key, value in msg.items() 
                             if key != 'aggregator_args' and logger.level != "DEBUG" }
                logger.info(f'\033[1mSending request\033[0m \n'
                            f'\t\t\t\t\t\033[1m To\033[0m: {str(cli)} \n'
                            f'\t\t\t\t\t\033[1m Request: \033[0m: Perform training with the arguments: '
                            f'{str(msg_print)} '
                            f'\n {5 * "-------------"}')

            time_start[cli] = time.perf_counter()
            self._reqs.send_message(msg, cli)  # send request to node

        # Recollect models trained
        self._training_replies[round_] = Responses([])
        while self.waiting_for_nodes(self._training_replies[round_]):
            # collect nodes responses from researcher request 'train'
            # (wait for all nodes with a ` while true` loop)
            # models_done = self._reqs.get_responses(look_for_commands=['train'])
            models_done = self._reqs.get_responses(look_for_commands=['train', 'error'], only_successful=False)
            for m in models_done.data():  # retrieve all models
                # (there should have as many models done as nodes)

                # manage error messages during training
                if m['command'] == 'error':
                    if m['extra_msg']:
                        logger.info(f"Error message received during training: {str(m['errnum'].value)} "
                                    f"- {str(m['extra_msg'])}")
                    else:
                        logger.info(f"Error message received during training: {str(m['errnum'].value)}")

                    faulty_node = m['node_id']  # remove the faulty node from the list

                    if faulty_node not in list(self._nodes):
                        logger.warning(f"Error message from {faulty_node} ignored, since this node is not part ot "
                                       f"the training any mode")
                        continue

                    self._nodes.remove(faulty_node)
                    continue

                # only consider replies for our request
                if m['researcher_id'] != environ['RESEARCHER_ID'] or \
                        m['job_id'] != self._id or m['node_id'] not in list(self._nodes):
                    continue

                rtime_total = time.perf_counter() - time_start[m['node_id']]

                # TODO : handle error depending on status
                if do_training:
                    logger.info(f"Downloading model params after training on {m['node_id']} - from {m['params_url']}")
                    try:
                        _, params_path = self.repo.download_file(m['params_url'],
                                                                 'node_params_' + str(uuid.uuid4()) + '.pt')
                    except FedbiomedRepositoryError as err:
                        logger.error(f"Cannot download model parameter from node {m['node_id']}, probably because Node"
                                     f" stops working (details: {err})")
                        return
                    params = self._training_plan.load(params_path, update_model=False)
                    model_params = params['model_params']
                    optimizer_args = params.get('optimizer_args')
                    encryption_factor = params.get('encryption_factor', None)
                else:
                    params_path = None
                    model_params = None
                    optimizer_args = None

                # TODO: could choose completely different name/structure for
                timing = m['timing']
                timing['rtime_total'] = rtime_total

                r = Responses({'success': m['success'],
                               'msg': m['msg'],
                               'dataset_id': m['dataset_id'],
                               'node_id': m['node_id'],
                               'params_path': params_path,
                               'params': model_params,
                               'optimizer_args': optimizer_args,
                               'sample_size': m["sample_size"],
                               'encryption_factor': encryption_factor,
                               'timing': timing})

                self._training_replies[round_].append(r)

        # return the list of nodes which answered because nodes in error have been removed
        return self._nodes

    def update_parameters(self,
                          params: dict = None,
                          filename: str = None,
                          is_model_params: bool = True,
                          variable_name: str = 'aggregated_params') -> Tuple[str, str]:
        """Updates global model aggregated parameters in `params`, by saving them to a file `filename` (unless it
        already exists), then upload file to the repository so that params are ready to be sent to the nodes for the
        next training round. If a `filename` is given (file exists) it has precedence over `params`.

        Args:
            params: data structure containing the new version of the aggregated parameters for this job,
            defaults to empty dictionary {}
            filename: path to the file containing the new version of the aggregated parameters for this job,
            defaults to None.
            is_model_params: whether params are models parameters or another value that must be sent
            through file exchange system. Defaults to True (argument are model parameters).
            variable_name:  name the filename with variable_name. Defaults to 'aggregated_prams'.

        Returns:
            the name of the parameter file
            the URL of the uploaded file
        """
        if params is None:
            params = {}
        try:
            if not filename:
                if not params:
                    raise ValueError('Bad arguments for update_parameters, filename or params is needed')
                filename = os.path.join(self._keep_files_dir, variable_name + str(uuid.uuid4()) + '.pt')

                # Needs to be copied in order to keep the object as np.ndarray
                params_to_save = copy.deepcopy(params)

                self._training_plan.save(filename, params_to_save)

            repo_response = self.repo.upload_file(filename)

            if is_model_params:
                # case where we are designing model parameter file
                self._repository_args['params_url'] = repo_response['file']
                self._model_params_file = filename
        except Exception as e:
            e = sys.exc_info()
            logger.error("Cannot update parameters - Error: " + str(e))
            sys.exit(-1)
        return filename, repo_response['file']

    def save_state(self, breakpoint_path: str) -> dict:
        """Creates current state of the job to be included in a breakpoint. Includes creating links to files included
        in the job state.

        Args:
            breakpoint_path: path to the existing breakpoint directory

        Returns:
            Job's current state for breakpoint
        """

        # Note: some state is passed to __init__() thus is not managed
        # as job state but as experiment state in current version
        state = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
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

    def load_state(self, saved_state: dict = None):
        """Load breakpoints state for a Job from a saved state

        Args:
            saved_state: breakpoint content
        """
        self._id = saved_state.get('job_id')
        self.update_parameters(filename=saved_state.get('model_params_path'))
        self._training_replies = self._load_training_replies(
            saved_state.get('training_replies'),
            self._training_plan.load
        )
        self._researcher_id = saved_state.get('researcher_id')

    @staticmethod
    def _save_training_replies(training_replies: Dict[int, Responses]) -> List[List[dict]]:
        """Extracts a copy of `training_replies` and prepares it for saving in breakpoint

        - strip unwanted fields
        - structure as list/dict, so it can be saved with JSON

        Args:
            training_replies: training replies of already executed rounds of the job

        Returns:
            Extract from `training_replies` formatted for breakpoint
        """
        converted_training_replies = []

        for round_ in training_replies.keys():
            training_reply = copy.deepcopy(training_replies[round_].data())
            # we want to strip some fields for the breakpoint
            for node in training_reply:
                del node['params']
            converted_training_replies.append(training_reply)

        return converted_training_replies

    @staticmethod
    def _load_training_replies(bkpt_training_replies: List[List[dict]],
                               func_load_params: Callable) -> Dict[int, Responses]:
        """Reads training replies from a formatted breakpoint file, and build a job training replies data structure .

        Args:
            bkpt_training_replies: Extract from training replies saved in breakpoint
            func_load_params: Function for loading parameters from file to training replies data structure

        Returns:
            Training replies of already executed rounds of the job
        """

        training_replies = {}
        for round_ in range(len(bkpt_training_replies)):
            loaded_training_reply = Responses(bkpt_training_replies[round_])
            # reload parameters from file params_path
            for node in loaded_training_reply:
                node['params'] = func_load_params(
                    node['params_path'], update_model=False)['model_params']

            training_replies[round_] = loaded_training_reply

        return training_replies

    def check_data_quality(self):
        """Does quality check by comparing datasets that have been found in different nodes. """

        data = self._data.data()
        # If there are more than two nodes ready for the job
        if len(data.keys()) > 1:

            # First check data types are same based on searched tags
            logger.info('Checking data quality of federated datasets...')

            data_types = []  # CSV, Image or default
            shapes = []  # dimensions
            dtypes = []  # variable types for CSV datasets

            # Extract features into arrays for comparison
            for feature in data.values():
                data_types.append(feature["data_type"])
                dtypes.append(feature["dtypes"])
                shapes.append(feature["shape"])

            if len(set(data_types)) > 1:
                raise FedbiomedDataQualityCheckError(
                    f'Different type of datasets has been loaded with same tag: {data_types}'
                )

            if data_types[0] == 'csv':
                if len(set([s[1] for s in shapes])) > 1:
                    raise FedbiomedDataQualityCheckError(
                        f'Number of columns of federated datasets do not match {shapes}.'
                    )

                dtypes_t = list(map(list, zip(*dtypes)))
                for t in dtypes_t:
                    if len(set(t)) > 1:
                        # FIXME: specifying a specific use case (in the condition above) should be avoided 
                        raise FedbiomedDataQualityCheckError(
                            f'Variable data types do not match in federated datasets {dtypes}'
                        )

            elif data_types[0] == 'images':
                shapes_t = list(map(list, zip(*[s[2:] for s in shapes])))
                dim_state = True
                for s in shapes_t:
                    if len(set(s)) != 1:
                        dim_state = False

                if not dim_state:
                    logger.error(f'Dimensions of the images in federated datasets \
                                 do not match. Please consider using resize. {shapes} ')

                if len(set([k[1] for k in shapes])) != 1:
                    logger.error(f'Color channels of the images in federated \
                                    datasets do not match. {shapes}')

            # If it is default MNIST dataset pass
            else:
                pass

        pass


class localJob:
    """Represents the entity that manage the training part. LocalJob is the version of Job but applied locally on a
    local dataset (thus not involving any network). It is only used to compare results to a Federated approach, using
    networks.
    """

    def __init__(self, dataset_path: str = None,
                 training_plan_class: str = 'MyTrainingPlan',
                 training_plan_path: str = None,
                 training_args: TrainingArgs = None,
                 model_args: dict = None):

        """
        Constructor of the class

        Args:
            dataset_path : The path where data is stored on local disk.
            training_plan_class: Name of the model class to use for training or model class.
            training_plan_path: path to file containing model code. Defaults to None.
            training_args: contains training parameters: lr, epochs, batch_size...
            model_args: contains output and input feature dimension.
        """

        self._id = str(uuid.uuid4())
        self._repository_args = {}
        self._localjob_training_args = training_args
        self._model_args = model_args
        self._training_args = TrainingArgs(training_args, only_required=False)
        self.dataset_path = dataset_path

        if training_args is not None:
            if training_args.get('test_on_local_updates', False) \
                    or training_args.get('test_on_global_updates', False):
                # if user wants to perform validation, display this message
                logger.warning("Cannot perform validation, not supported for LocalJob")

        # handle case when model is in a file
        if training_plan_path is not None:
            try:
                model_module = os.path.basename(training_plan_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(training_plan_path))

                module = importlib.import_module(model_module)
                tr_class = getattr(module, training_plan_class)
                self._training_plan = tr_class()
                sys.path.pop(0)

            except Exception as e:
                e = sys.exc_info()
                logger.critical("Cannot import class " + training_plan_class + " from path " +
                                training_plan_path + " - Error: " + str(e))
                sys.exit(-1)
        else:

            # create/save model instance
            if inspect.isclass(training_plan_class):
                self._training_plan = training_plan_class()
            else:
                self._training_plan = training_plan_class

        self._training_plan.post_init(model_args=self._model_args,
                                      training_args=self._training_args)

    @property
    def training_plan(self):
        return self._training_plan

    @property
    def model(self):
        return self._training_plan.model()

    @property
    def training_args(self):
        return self._localjob_training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self._localjob_training_args = training_args

    def start_training(self):
        """Sends training task to nodes and waits for the responses"""

        for i in self._training_plan._dependencies:
            exec(i, globals())

        is_failed = False
        error_message = ''

        # Run the training routine
        if not is_failed:
            results = {}
            try:
                self._training_plan.set_dataset_path(self.dataset_path)
                data_manager = self._training_plan.training_data()
                tp_type = self._training_plan.type()
                data_manager.load(tp_type=tp_type)
                train_loader, test_loader = data_manager.split(test_ratio=0)
                self._training_plan.training_data_loader = train_loader
                self._training_plan.testing_data_loader = test_loader
                self._training_plan.training_routine()
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model in job : " + str(e)

        if not is_failed:
            try:
                # TODO : should test status code but not yet returned
                # by upload_file
                filename = environ['TMP_DIR'] + '/local_params_' + str(uuid.uuid4()) + '.pt'
                self._training_plan.save(filename, results)
            except Exception as e:
                is_failed = True
                error_message = "Cannot write results: " + str(e)

        if error_message != '':
            logger.error(error_message)
