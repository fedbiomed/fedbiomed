# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import copy
import os
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from fedbiomed.common.exceptions import FedbiomedRepositoryError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_unique_link, create_unique_file_link
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.federated_workflows._job import Job


class TrainingJob(Job):
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

        super().__init__(
            reqs,
            nodes,
            training_plan_class,
            training_plan_path,
            training_args,
            data,
            keep_files_dir
        )
        self._model_args = model_args
        self._model_file = None  # path to local file containing model code
        self._model_params_file = ""  # path to local file containing current version of aggregated params
        self._aggregator_args = None

        # Model is mandatory
        if self._training_plan_class is None:
            mess = "Missing training plan class name or instance in Job arguments"
            logger.critical(mess)
            raise NameError(mess)

        self._training_plan = self.training_plan.load_training_plan_from_file(
            self._keep_files_dir,
            self._training_plan_module,
            self._training_plan_name)

        self._training_plan.post_init(model_args={} if self._model_args is None else self._model_args,
                                      training_args=self._training_args)

        # Save model parameters to a local file and upload it to the remote repository.
        # The filename and remote url are assigned to attributes through this call.
        try:
            self.update_parameters()
        except SystemExit:
            return

        # Validate fields in each argument
        self.validate_minimal_arguments(self._repository_args,
                                        ['training_plan_url', 'training_plan_class', 'params_url'])

    @property
    def aggregator_args(self):
        return self._aggregator_args

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
                try:
                    filename = os.path.join(self._keep_files_dir, f"{arg_name}_{uuid.uuid4()}.mpk")
                    Serializer.dump(aggr_param, filename)
                    url = self.repo.upload_file(filename)["file"]
                except Exception as exc:
                    logger.critical("Failed to export %s to local file and upload it: %s", arg_name, exc)
                    sys.exit(-1)
                args_thr_msg[node_id][arg_name]['filename'] = filename  # path to the file with the parameters
                args_thr_msg[node_id][arg_name]['url'] = url

        return args_thr_msg

    def start_nodes_training_round(
        self,
        round_: int,
        aggregator_args_thr_msg: Dict[str, Dict[str, Any]],
        aggregator_args_thr_files: Dict[str, Dict[str, Any]],
        secagg_arguments: Union[Dict, None] = None,
        do_training: bool = True,
        optim_aux_var: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
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
            optim_aux_var: Auxiliary variables of the researcher-side Optimizer, if any.
                Note that such variables may only be used if both the Experiment and node-side training plan
                hold a declearn-based [Optimizer][fedbiomed.common.optimizers.Optimizer], and their plug-ins
                are coherent with each other as to expected information exchange.
        """

        # Assign empty dict to secagg arguments if it is None
        if secagg_arguments is None:
            secagg_arguments = {}

        headers = {
            'researcher_id': self._researcher_id,
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
            'aggregator_args': {},
            'aux_var_urls': None,
        }

        msg = {**headers, **self._repository_args}
        time_start = {}

        # pass heavy aggregator params through file exchange system
        self.upload_aggregator_args(aggregator_args_thr_msg, aggregator_args_thr_files)

        # Upload optimizer auxiliary variables, when there are.
        if do_training and optim_aux_var:
            aux_url_shared, aux_url_bynode = (
                self.upload_agg_optimizer_aux_var(optim_aux_var)
            )
        else:
            aux_url_shared = None
            aux_url_bynode = {}

        for cli in self._nodes:
            msg['dataset_id'] = self._data.data()[cli]['dataset_id']
            cli_aux_urls = (aux_url_shared, aux_url_bynode.get(cli, None))
            msg['aux_var_urls'] = [url for url in cli_aux_urls if url] or None

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

                # manage training failure for this job
                if not m['success']:
                    logger.error(f"Training failed for node {m['node_id']}: {m['msg']}")
                    self._nodes.remove(m['node_id'])  # remove the faulty node from the list
                    continue

                rtime_total = time.perf_counter() - time_start[m['node_id']]

                if do_training:
                    logger.info(f"Downloading model params after training on {m['node_id']} - from {m['params_url']}")
                    try:
                        _, params_path = self.repo.download_file(m["params_url"], f"node_params_{uuid.uuid4()}.mpk")
                    except FedbiomedRepositoryError as err:
                        logger.error(f"Cannot download model parameter from node {m['node_id']}, probably because Node"
                                     f" stops working (details: {err})")
                        return
                    results = Serializer.load(params_path)
                    params = results["model_weights"]
                    optimizer_args = results.get("optimizer_args")
                    optim_aux_var = results.get("optim_aux_var", {})
                    encryption_factor = results.get('encryption_factor', None)
                else:
                    params_path = None
                    params = None
                    optimizer_args = None
                    encryption_factor = None

                # TODO: could choose completely different name/structure for
                timing = m['timing']
                timing['rtime_total'] = rtime_total

                response = Responses({
                    'success': m['success'],
                    'msg': m['msg'],
                    'dataset_id': m['dataset_id'],
                    'node_id': m['node_id'],
                    'params_path': params_path,
                    'params': params,
                    'optimizer_args': optimizer_args,
                    'optim_aux_var': optim_aux_var,
                    'sample_size': m["sample_size"],
                    'encryption_factor': encryption_factor,
                    'timing': timing,
                })
                self._training_replies[round_].append(response)

        # return the list of nodes which answered because nodes in error have been removed
        return self._nodes

    def upload_agg_optimizer_aux_var(
        self,
        aux_var: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[str], Dict[uuid.UUID, str]]:
        """Upload auxiliary variables emitted by a researcher-side Optimizer.

        Args:
            aux_var: Dict of auxiliary variables emitted by an Optimizer held
                by the researcher, that are to be uploaded after having been
                structured into multiple files, to avoid information leakage
                as well as content redundancy.

        Returns:
            url_shared: url of a file containing auxiliary variables shared
                across all nodes, or None (in the absence of such information).
            url_bynode: dict mapping urls of files containing node-specific
                auxiliary variables to the nodes' id (a missing `nodes` key
                indicates that this node has no such information to receive).

        !!!info "Note":
            The use of both a shared URL and node-specific one is merely a
            way to reduce communication costs by uploading only once the
            information that is to be downloaded by each and every node.
        """
        # Split the information between shared and node-wise dictionaries.
        aux_shared, aux_bynode = self._prepare_agg_optimizer_aux_var(
            aux_var=aux_var, nodes=list(self._nodes)
        )
        # Upload the shared information that all nodes will download.
        if aux_shared:
            path = os.path.join(
                self._keep_files_dir, f"aux_var_shared_{uuid.uuid4()}.mpk"
            )
            Serializer.dump(aux_shared, path)
            url_shared = self.repo.upload_file(path)["file"]
        else:
            url_shared = None
        # Upload the node-specific information, with node-specific urls.
        url_bynode = {}  # type: Dict[uuid.UUID, str]
        for node_id, node_aux in aux_bynode.items():
            if not node_aux:
                continue
            path = os.path.join(
                self._keep_files_dir,
                f"aux_var_node_{node_id}_{uuid.uuid4()}.mpk"
            )
            Serializer.dump(node_aux, path)
            url_bynode[node_id] = self.repo.upload_file(path)["file"]
        # Return the urls of the uploaded files.
        return url_shared, url_bynode

    @staticmethod
    def _prepare_agg_optimizer_aux_var(
        aux_var: Dict[str, Dict[str, Any]],
        nodes: List[uuid.UUID],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[uuid.UUID, Dict[str, Dict[str, Any]]],
    ]:
        """Collect and structure researcher-side Optimizer auxiliary variables.

        Args:
            aux_var: Auxiliary variables with to structure into multiple dicts,
                from `{mod_name: (shared_dict | {node_id: node_dict})}` to
                `{mod_name: shared_dict}` & `{node_id: {mod_name: node_dict}}`.
            nodes: Ids of the nodes to whom auxiliary variables should be
                sent. This is used to drop information of non-participating
                nodes.

        Returns:
            aux_shared: Dict containing auxiliary variables that are shared
                across all nodes, with `{mod_name: shared_dict}` format.
            aux_bynode: Dict containing node-wise dicts of node-specific
                auxiliary variables, with `{node_id: {mod_name: node_dict}}`
                format.
        """
        aux_shared = {}  # type: Dict[str, Dict[str, Any]]
        aux_bynode = {}  # type: Dict[uuid.UUID, Dict[str, Dict[str, Any]]]
        # Iterate over nodes and plug-in-module-wise auxiliary variables.
        for node_id in nodes:
            aux_bynode[node_id] = {}
            for mod_name, mod_info in aux_var.items():
                # Case of node-specfic information.
                if node_aux := mod_info.get(str(node_id)):
                    aux_bynode[node_id][mod_name] = node_aux
                # Case of global information shared with all nodes.
                elif mod_name not in aux_shared:
                    aux_shared[mod_name] = mod_info
        # Return the restructured auxiliary variables dicts.
        return aux_shared, aux_bynode

    def extract_received_optimizer_aux_var_from_round(
        self,
        round_id: int,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Restructure the received auxiliary variables (if any) from a round.

        Args:
            round_id: Index of the round, replies from which to parse through.

        Returns:
            Dict of auxiliary variables, collating node-wise information, with
            format `{mod_name: {node_id: node_dict}}`.
        """
        aux_var = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
        for reply in self.training_replies[round_id]:
            node_id = reply["node_id"]
            node_av = reply.get("optim_aux_var", {})
            for module, params in node_av.items():
                aux_var.setdefault(module, {})[node_id] = params
        return aux_var

    def update_parameters(
        self,
        params: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Save and upload global model parameters, optionally after updating them.

        This method is designed to save and upload the parameters of the wrapped
        training plan instance. It may also be used to update these parameters
        prior to their upload, whether based on provided in-memory values or on
        a pre-exported dump file.

        Args:
            params: data structure containing the new version of the aggregated parameters for this job,
            defaults to empty dictionary {}
            filename: path to the file containing the new version of the aggregated parameters for this job,
            defaults to None.
            is_model_params: whether params are models parameters or another value that must be sent
            through file exchange system. Defaults to True (argument are model parameters).
            variable_name:  name the filename with variable_name. Defaults to 'aggregated_prams'.

            params: Optional dict storing new aggregated parameters that are to
                be assigned to this job's training plan's model.
                If None, export and upload the current model parameters.
            filename: Optional path to a pre-existing file containing the
                aggregated parameters to load an upload.
                If `params` is not None, `filename` has to be None.

        Returns:
            filename: path to the local parameters file
            url: url at which the file was uploaded

        Raises:
            ValueError: if both `params` and `filename` are provoided: these parameters are mutually-exclusive.

        !!! info "Notes":
            * The path to the created and/or uploaded file is stored under the `_model_params_file` attribute,
              that is updated by this method.
            * The url of the uploaded file is stored under the `_repository_args["params_url"]` attribute,
              that is also updated by this method.

        !!! warning "Warning":
            * The `params` and `filename` parameters are mutually-exclusive.
        """
        try:
            if params and filename:
                raise ValueError("'update_parameters' received both filename and params: only one may be used.")
            # Case when uploading a pre-existing file: load the parameters.
            if filename:
                params = Serializer.load(filename)["model_weights"]
                self._training_plan.set_model_params(params)
            # Case when exporting current parameters: create a local dump file.
            else:
                # Case when uploading the current parameters: gather them.
                if params is None:
                    params = self._training_plan.get_model_params()
                # Case when uploading a new set of parameters: assign them.
                else:
                    self._training_plan.set_model_params(params)
                # At any rate, create a local dump file.
                filename = os.path.join(self._keep_files_dir, f"aggregated_params_{uuid.uuid4()}.mpk")
                params_dump = {
                    "researcher_id": self._researcher_id,
                    "model_weights": params,
                }
                Serializer.dump(params_dump, filename)
            # Upload the file and record its local and remote locations.
            self._model_params_file = filename
            repo_response = self.repo.upload_file(filename)
            self._repository_args["params_url"] = url = repo_response["file"]
            # Return the local path and remote url to the file.
            return filename, url
        # Log exceptions and trigger a system exit if one is raised.
        except Exception:
            exc = sys.exc_info()
            logger.error("'Job.update_parameters' failed with error: %s", exc)
            sys.exit(-1)

    def save_state(self, breakpoint_path: str) -> dict:
        """Creates current state of the job to be included in a breakpoint.

        Includes creating links to files included in the job state.

        Args:
            breakpoint_path: path to the existing breakpoint directory

        Returns:
            Job's current state for breakpoint
        """

        # Note: some state is passed to __init__() thus is not managed
        # as job state but as experiment state in current version
        state = super().save_state(breakpoint_path)

        state.update({'model_params_path': self._model_params_file})

        state['model_params_path'] = create_unique_link(
            breakpoint_path, 'aggregated_params_current', '.mpk',
            os.path.join('../..', os.path.basename(state["model_params_path"]))
        )

        for round_replies in state['training_replies']:
            for response in round_replies:
                node_params_path = create_unique_file_link(
                    breakpoint_path, response['params_path']
                )
                response['params_path'] = node_params_path

        return state

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """Load breakpoints state for a Job from a saved state

        Args:
            saved_state: breakpoint content
        """
        super().load_state(saved_state)
        self.update_parameters(filename=saved_state.get("model_params_path"))

    @staticmethod
    def _save_training_replies(training_replies: Dict[int, Responses]) -> List[List[Dict[str, Any]]]:
        """Extracts a copy of `training_replies` and prepares it for saving in breakpoint

        - strip unwanted fields
        - structure as list/dict, so it can be saved with JSON

        Args:
            training_replies: training replies of already executed rounds of the job

        Returns:
            Extract from `training_replies` formatted for breakpoint
        """
        converted_training_replies = super()._save_training_replies(training_replies)

        for round_ in training_replies.keys():
            training_reply = copy.deepcopy(training_replies[round_].data())
            # we want to strip some fields for the breakpoint
            for node in training_reply:
                del node['params']
            converted_training_replies.append(training_reply)

        return converted_training_replies

    @staticmethod
    def _load_training_replies(bkpt_training_replies: List[List[dict]]) -> Dict[int, Responses]:
        """Reads training replies from a formatted breakpoint file, and build a job training replies data structure .

        Args:
            bkpt_training_replies: Extract from training replies saved in breakpoint

        Returns:
            Training replies of already executed rounds of the job
        """

        training_replies = super()._load_training_replies(bkpt_training_replies)
        if not bkpt_training_replies:
            logger.warning("No Replies has been found in this breakpoint")
        for round_ in range(len(bkpt_training_replies)):
            loaded_training_reply = Responses(bkpt_training_replies[round_])
            # reload parameters from file params_path
            for node in loaded_training_reply:
                node["params"] = Serializer.load(node["params_path"])["model_weights"]
            training_replies[round_] = loaded_training_reply

        return training_replies

