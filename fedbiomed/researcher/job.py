"""Manage the training part of the experiment."""

import atexit
import copy
import os
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import declearn
from declearn.model.api import NumpyVector

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.exceptions import FedbiomedRepositoryError
from fedbiomed.common.logger import logger
from fedbiomed.common.repository import Repository
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TrainingPlan
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import (
    create_unique_file_link,
    create_unique_link,
)
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses


class Job:
    """
    Represents the entity that manage the training part at  the nodes level

    Starts a message queue, loads python model file created by researcher (through
    [`training_plans`][fedbiomed.common.training_plans]) and saves the loaded model in a temporary file
    (under the filename '<TEMP_DIR>/training_plan_<random_id>.py').

    """

    def __init__(
        self,
        training_plan: TrainingPlan,
        training_args: TrainingArgs,
        model_args: Dict[str, Any],
        data: FederatedDataSet,
        reqs: Optional[Requests] = None,
        nodes: Optional[List[uuid.UUID]] = None,
        keep_files_dir: Optional[str] = None,
    ) -> None:
        """Instantiate a Job.

        Args:
            training_plan: Training plan instance that wraps training.
            training_args: Contains training parameters; lr, epochs, batch_size.
            model_args: Contains output and input feature dimension
            nodes: A list of node_id specifying the nodes used for training
            data: Federated datasets
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        Raises:
            NameError: If model is not defined or if the class can not to be inspected
        """
        # Assign input parameters as private attributes.
        self._training_plan = training_plan
        self._training_args = training_args
        self._model_args = model_args
        self._data = data
        self._reqs = Requests() if reqs is None else reqs
        self._nodes = nodes or []
        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ["TMP_DIR"])
            atexit.register(
                lambda: shutil.rmtree(self._keep_files_dir)
            )  # remove directory
            # when script ends running (replace
            # `with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as self._keep_files_dir: `)
        # Create additional attributes: ids and data containers.
        self._id = str(uuid.uuid4())  # creating a unique job id
        self._researcher_id = environ["RESEARCHER_ID"]
        self._repository_args = {}
        self._training_replies = {}  # type: Dict[int, Responses]
        # Set up a repository to upload and download files.
        self.repo = Repository(
            environ["UPLOADS_URL"], self._keep_files_dir, environ["CACHE_DIR"]
        )
        # Assign paths to local JSON dumps of the training plan and its params.
        self._training_plan_file = os.path.join(
            self._keep_files_dir, f"training_plan_{uuid.uuid4()}.json"
        )
        self._model_params_file = os.path.join(
            self._keep_files_dir, f"aggregated_params_init_{uuid.uuid4()}.json"
        )
        # Check dataset quality
        self.check_data_quality()
        # Finalize the training plan's instantiation.
        self._training_plan.post_init(
            model_args=self._model_args, training_args=self._training_args
        )
        # Dump to file and upload the training plan instance.
        self._training_plan.save_to_json(self._training_plan_file)
        repo_response = self.repo.upload_file(self._training_plan_file)
        self._repository_args["training_plan_url"] = repo_response["file"]
        # Dump to file and upload the training plan's weights.
        self._training_plan.save_weights(self._model_params_file)
        repo_response = self.repo.upload_file(self._model_params_file)
        self._repository_args["params_url"] = repo_response["file"]

    @property
    def training_plan(self) -> TrainingPlan:
        return self._training_plan

    @property
    def training_plan_file(self) -> str:
        return self._training_plan_file

    @property
    def requests(self) -> Requests:
        return self._reqs

    @property
    def nodes(self) -> List[uuid.UUID]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[uuid.UUID]) -> None:
        self._nodes = nodes

    @property
    def training_replies(self) -> Dict[int, Responses]:
        return self._training_replies

    @property
    def training_args(self) -> Dict[str, Any]:
        return self._training_args.dict()

    @training_args.setter
    def training_args(self, training_args: TrainingArgs) -> None:
        self._training_args = training_args

    def check_training_plan_is_approved_by_nodes(self) -> Responses:
        """Checks whether the training plan is approved or not.

        This method sends `training-plan-status` request to the nodes.
        It should be run before running experiment.
        So, researchers can find out if their model has been approved
        """
        message = {
            "researcher_id": self._researcher_id,
            "job_id": self._id,
            "training_plan_url": self._repository_args["training_plan_url"],
            "command": "training-plan-status",
        }
        responses = Responses([])
        replied_nodes = []
        node_ids = self._data.node_ids()
        # Send message to each node that has been found after dataset search request
        for cli in node_ids:
            logger.info(
                f"Sending request to node {cli} to check whether the "
                "training plan is approved or not."
            )
            self._reqs.send_message(message, str(cli))
        # Wait for responses
        for resp in self._reqs.get_responses(
            look_for_commands=["training-plan-status"], only_successful=False
        ):
            responses.append(resp)
            replied_nodes.append(resp.get("node_id"))
            if resp.get("success"):
                if resp.get("approval_obligation"):
                    if resp.get("status") == TrainingPlanApprovalStatus.APPROVED.value:
                        logger.info(
                            f'Training plan has been approved by the node: {resp.get("node_id")}'
                        )
                    else:
                        logger.warning(
                            f'Training plan has NOT been approved by the node: {resp.get("node_id")}.'
                            + f'Training plan status : {resp.get("status")}'
                        )
                else:
                    logger.info(
                        f'Training plan approval is not required by the node: {resp.get("node_id")}'
                    )
            else:
                logger.warning(
                    f"Node : {resp.get('node_id')} : {resp.get('msg')}"
                )
        # Get the nodes that haven't replied training-plan-status request
        non_replied_nodes = list(set(node_ids) - set(replied_nodes))
        if non_replied_nodes:
            logger.warning(
                "Request for checking training plan status hasn't been replied"
                f"by the nodes: {non_replied_nodes}. You might get an error"
                "while running your experiment."
            )
        return responses

    def waiting_for_nodes(self, responses: Responses) -> bool:
        """Verify if all nodes involved in the job have responded.

        Args:
            responses: Container for received replies from nodes.

        Returns:
            waiting: Whether some nodes are still to respond, or all
                are present in the `responses` object.
        """
        try:
            nodes_done = set(responses.dataframe()["node_id"])
        except KeyError:
            nodes_done = set()
        return nodes_done != set(self._nodes)

    def start_nodes_training_round(
            self,
            round_idx: int,
            do_training: bool = True,
            aux_vars: Optional[Dict[str, Dict[str, Any]]] = None,
        ) -> List[uuid.UUID]:
        """Sends training request to nodes and waits for the responses

        Args:
            round_idx: Index of the round being currently performed.
            do_training: Whether to conduct training in this round,
                or only perform validation.
            aux_vars: Optional optimizer auxiliary variables to be shared with
                clients, formatted as a {module_name: variables} dict, where
                'variables' may either be a dict of parameters, or a mapping
                of node-and-dataset-wise parameters associated to an id made
                from node-id and dataset-id.
        """
        # Send training instructions to nodes.
        time_start = self._send_training_requests(do_training, aux_vars)
        # Collect and process nodes' replies.
        self._training_replies[round_idx] = Responses([])
        while self.waiting_for_nodes(self._training_replies[round_idx]):
            responses = self._reqs.get_responses(
                look_for_commands=["train", "error"], only_successful=False
            )
            for msg in responses.data():
                # only consider replies to our request
                if (
                    msg["researcher_id"] != self._researcher_id
                    or msg["job_id"] != self._id
                    or msg["node_id"] not in list(self._nodes)
                ):
                    continue
                # report error messages during training
                # and remove the associated nodes from `self._nodes`
                if "errnum" in msg:  # TODO: need a stronger filter
                    err = (
                        f"Error message received during training: "
                        f"{msg['errnum'].value}"
                    )
                    if msg["extra_msg"]:
                        err += f" - {msg['extra_msg']}"
                    logger.info(err)
                    faulty_node = msg["node_id"]
                    self._nodes.remove(faulty_node)
                    continue
                # unpack and complete timing information about the round
                timing = msg["timing"]
                timing["rtime_total"] = (
                    time.perf_counter() - time_start[msg["node_id"]]
                )
                # download and load into memory the updated model parameters
                # and optional optimizer auxiliary variables
                if do_training:
                    params_path, params, aux_vars_path, aux_vars = (
                        self._download_node_parameters(msg)
                    )
                    if params is None:
                        logger.info(
                            f"Removing {msg['node_id']} due to failure to "
                            "download sent model parameters."
                        )
                        self._nodes.remove(msg['node_id'])
                        continue
                else:
                    params_path = params = aux_vars_path = aux_vars = None
                # package the received information and record it
                response = {
                        "success": msg["success"],
                        "msg": msg["msg"],
                        "dataset_id": msg["dataset_id"],
                        "node_id": msg["node_id"],
                        "params_path": params_path,
                        "params": params,
                        "aux_vars_path": aux_vars_path,
                        "aux_vars": aux_vars,
                        "timing": timing,
                }
                self._training_replies[round_idx].append(response)
        # return the list of nodes which answered (some may have been removed).
        return self._nodes

    def _send_training_requests(
            self,
            do_training: bool = True,
            aux_vars: Optional[Dict[str, Dict[str, Any]]] = None,
        ) -> Dict[uuid.UUID, float]:
        """Send training requests to nodes for a round.

        Args:
            do_training: Whether to conduct training in this round,
                or only perform validation.
            aux_vars: Optional optimizer auxiliary variables to be shared with
                clients, formatted as a {module_name: variables} dict, where
                'variables' may either be a dict of parameters, or a mapping
                of node-and-dataset-wise parameters associated to an id made
                from node-id and dataset-id.

        Returns:
            time_start: Dict mapping the time at which the training request
                were sent to the nodes' ids.
        """
        # Prepare shared information to send to nodes.
        msg = {
            "command": "train",
            "job_id": self._id,
            "researcher_id": self._researcher_id,
            "training_plan_url": self._repository_args["training_plan_url"],
            "params_url": self._repository_args["params_url"],
            "training_args": self._training_args.dict(),
            "model_args": self._model_args,
            "training": do_training,
        }
        # Iteratively send training requests to the nodes.
        time_start = {}
        for cli in self._nodes:
            # Add node-specific information to the message.
            msg["training_data"] = {
                cli: [ds["dataset_id"] for ds in self._data.data()[cli]]
            }
            if aux_vars is None:
                msg["aux_vars_url"] = None
            else:
                # Refine auxiliary variables, as they can be node-specific.
                c_aux_vars = {}
                for dst in self._data.data()[cli]:
                    uid = f"{cli}/{dst['dataset_id']}"
                    c_aux_vars[uid] = {
                        module: config.get(uid, config)
                        for module, config in aux_vars.items()
                    }
                # Upload the auxiliary variables and add their URL to msg.
                msg["aux_vars_url"] = self.upload_file(c_aux_vars, "aux_vars")
            # Log about what is being done.
            if not do_training:
                logger.info(
                    f"\033[1mSending request\033[0m \n"
                    f"\t\t\t\t\t\033[1m To\033[0m: {cli} \n"
                    f"\t\t\t\t\t\033[1m Request: \033[0m: Perform validation "
                    f"on aggregated parameters \n {5 * '-------------'}"
                )
            else:
                logger.info(
                    f"\033[1mSending request\033[0m \n"
                    f"\t\t\t\t\t\033[1m To\033[0m: {cli} \n"
                    f"\t\t\t\t\t\033[1m Request: \033[0m: Perform training "
                    f"with the arguments: {msg} \n {5 * '-------------'}"
                )
            # Record that current time and send the request to the node.
            time_start[cli] = time.perf_counter()
            self._reqs.send_message(msg, str(cli))
        # Return the dict storing time information.
        return time_start

    def upload_parameters(self) -> str:
        """Save to file and upload the wrapped training plan's parameters.

        Update some private attributes of the Job as a result:
        - `_model_params_file` records the path to the created dump file
        - `_repository_args["params_url"]` records the URL at which it
          was made available to downlaod by nodes

        Returns:
            params_path: Path to the created model parameters JSON dump file.

        Raises:
            SystemExit: if the operation fails.
        """
        try:
            # Dump the training plan parameters to a local JSON file.
            filename = os.path.join(
                self._keep_files_dir,
                f"aggregated_params_{uuid.uuid4()}.json",
            )
            self._training_plan.save_weights(filename)
            # Upload the file to the remote files repository.
            repo_response = self.repo.upload_file(filename)
            # Update attributes: local path and remote URL of the file.
            self._repository_args["params_url"] = repo_response["file"]
            self._model_params_file = filename
        except Exception as exc:
            logger.error(f"Cannot update parameters - Error: {exc}")
            sys.exit(-1)
        return self._model_params_file

    def upload_file(
            self,
            content: Dict[str, Any],
            basename: str = "",
        ) -> str:
        """Upload given content to the file server as a JSON file.

        Args:
            content: JSON-serializable data to write to a file and
                upload to the remote file repository.
            basename: Prefix to the created local dump file's name.
                Also used to provide context in logged messages.

        Returns:
            url: URL at which the file has been uploaded.

        Raises:
            SystemExit: if the operation fails.
        """
        try:
            filename = os.path.join(
                self._keep_files_dir,
                f"{basename}_{uuid.uuid4()}.json",
            )
            declearn.utils.json_dump(content, filename)
            repo_response = self.repo.upload_file(filename)
            return repo_response["file"]  # type: ignore
        except Exception as exc:
            logger.error(f"Cannot update {basename} - Error: {exc}")
            sys.exit(-1)

    def _download_node_parameters(
            self,
            msg: Dict[str, Any]
        ) -> Tuple[
            Optional[str],
            Optional[NumpyVector],
            Optional[str],
            Optional[Dict[str, Any]]
        ]:
        """Download some node-emitted model parameters and opt. aux. vars.

        Args:
            msg: Response to a TraingRequest send by this Job.

        Returns:
            params_path: Path to the local copy of the model parameters file.
                May be None in case of download failure.
            params: NumpyVector storing the downloaded model parameters.
                May be None in case of downlaod failure.
            aux_vars_path: Path to the local copy of the optimizer auxiliary
                variables file, if any were communicated by the node.
            aux_vars: Dict storing the received auxiliary variables, if any.
        """
        # Download model parameters.
        logger.info(
            f"Downloading model params after training on node {msg['node_id']}"
            f" - from {msg['params_url']}"
        )
        try:
            _, params_path = self.repo.download_file(
                msg["params_url"], f"node_params_{uuid.uuid4()}.json"
            )
        except FedbiomedRepositoryError as err:
            logger.error(
                "Failed to download model parameters from node "
                f"{msg['node_id']}, details: {err}"
            )
            return (None, None, None, None)
        params = self._training_plan.load_weights(params_path, assign=False)
        # Download optimizer auxiliary variables, if any.
        if not msg["aux_vars_url"]:
            return params_path, params, None, None
        logger.info(
            "Downloading optimizer auxiliary variables after training on "
            f"nose {msg['node_id']} - from {msg['aux_vars_url']}"
        )
        try:
            _, aux_vars_path = self.repo.download_file(
                msg["aux_vars_url"], f"node_aux_vars_{uuid.uuid4()}.json",
            )
        except FedbiomedRepositoryError as err:
            logger.error(
                "Failed to download auxiliary variables from node "
                f"{msg['node_id']}, details: {err}"
            )
            return params_path, params, None, None
        aux_vars = declearn.utils.json_load(aux_vars_path)
        return params_path, params, aux_vars_path, aux_vars

    def save_state(self, breakpoint_path: str) -> Dict[str, Any]:
        """Creates current state of the job to be included in a breakpoint.

        Includes creating links to files included in the job state.

        Args:
            breakpoint_path: path to the existing breakpoint directory

        Returns:
            Job's current state for breakpoint
        """
        # Note: some state is passed to __init__() thus is not managed
        # as job state but as experiment state in current version
        params_path = create_unique_link(
            breakpoint_path,
            "aggregated_params_current",
            ".json",
            os.path.join("..", os.path.basename(self._model_params_file)),
        )
        state = {
            "researcher_id": self._researcher_id,
            "job_id": self._id,
            "model_params_path": params_path,
            "training_replies": self._pack_training_replies(),
        }
        for round_replies in state["training_replies"]:
            for reply in round_replies:
                reply["params_path"] = create_unique_file_link(
                    breakpoint_path, reply["params_path"]
                )
                if reply["aux_vars_path"] is not None:
                    reply["aux_vars_path"] = create_unique_file_link(
                        breakpoint_path, reply["aux_vars_path"]
                    )
        return state

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """Load breakpoints state for a Job from a saved state

        Args:
            saved_state: breakpoint content
        """
        self._researcher_id = saved_state.get("researcher_id")
        self._id = saved_state["job_id"]
        # Reload the history of training replies.
        self._training_replies = self._unpack_training_replies(
            saved_state["training_replies"]
        )
        # Reload the current model parameters.
        params_path = saved_state["model_params_path"]  # type: str
        self._training_plan.load_weights(params_path)

    def _pack_training_replies(self) -> List[List[dict]]:
        """Extract a copy of `self._training_replies` for saving in breakpoint.

        - strip unwanted fields
        - structure as list/dict, so it can be saved with JSON

        Returns:
            Extract from `self._training_replies` formatted for breakpoint
        """
        converted_training_replies = []
        for responses in self._training_replies.values():
            resp_dict = copy.deepcopy(responses.data())
            # we want to strip some fields for the breakpoint
            for reply in resp_dict:
                reply.pop("params")
                reply.pop("aux_vars")
            converted_training_replies.append(resp_dict)
        return converted_training_replies

    def _unpack_training_replies(
        self,
        bkpt_training_replies: List[List[dict]],
    ) -> Dict[int, Responses]:
        """Unpack training replies recovered from a formatted breakpoint file.

        Args:
            bkpt_training_replies: Extract from training replies saved
                in breakpoint (using `self._pack_training_replies`).

        Returns:
            Training replies of already executed rounds of the job.
        """
        training_replies = {}
        for round_idx, responses in enumerate(bkpt_training_replies):
            loaded_responses = Responses(responses)
            # reload parameters from file params_path
            for reply in loaded_responses:
                reply["params"] = self._training_plan.load_weights(
                    reply["params_path"], assign=False
                )
                reply["aux_vars"] = (
                    declearn.utils.json_load(reply["aux_vars_path"])
                    if reply["aux_vars_path"] else None
                )
            training_replies[round_idx] = loaded_responses
        return training_replies

    def check_data_quality(self) -> None:
        """Does quality check by comparing datasets that have been found in different nodes."""

        data = self._data.data()
        # If there are more than two nodes ready for the job
        if len(data.keys()) > 1:

            # First check data types are same based on searched tags
            logger.info("Checking data quality of federated datasets...")

            data_types = []  # CSV, Image or default
            shapes = []  # dimensions
            dtypes = []  # variable types for CSV datasets

            # Extract features into arrays for comparison
            for data_list in data.items():
                for feature in data_list[1]:
                    data_types.append(feature["data_type"])
                    dtypes.append(feature["dtypes"])
                    shapes.append(feature["shape"])

            assert (
                len(set(data_types)) == 1
            ), f"Different type of datasets has been loaded with same tag: {data_types}"

            if data_types[0] == "csv":
                assert (
                    len(set(s[1] for s in shapes)) == 1
                ), f"Number of columns of federated datasets do not match {shapes}."

                dtypes_t = list(map(list, zip(*dtypes)))
                for t in dtypes_t:
                    assert (
                        len(set(t)) == 1
                    ), f"Variable data types do not match in federated datasets {dtypes}"

            elif data_types[0] == "images":
                shapes_t = list(map(list, zip(*[s[2:] for s in shapes])))
                dim_state = True
                for s in shapes_t:
                    if len(set(s)) != 1:
                        dim_state = False

                if not dim_state:
                    logger.error(
                        f"Dimensions of the images in federated datasets \
                                 do not match. Please consider using resize. {shapes} "
                    )

                if len(set(k[1] for k in shapes)) != 1:
                    logger.error(
                        f"Color channels of the images in federated \
                                    datasets do not match. {shapes}"
                    )
            # Else: it is default MNIST dataset pass, no action required


class LocalJob:
    """Represents the entity that manages the training part.

    LocalJob is a version of Job applied locally, on a local dataset
    (thus not involving any network).

    It is only used to compare results to a Federated approach.
    """

    def __init__(
        self,
        dataset_path: str = None,
        training_plan: TrainingPlan = None,
        training_args: TrainingArgs = None,
        model_args: dict = None,
    ) -> None:
        """Constructor of the local job.

        Args:
            dataset_path : The path where data is stored on local disk.
            training_plan: Training plan instance used for training.
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
            if training_args.get(
                "test_on_local_updates", False
            ) or training_args.get("test_on_global_updates", False):
                # if user wants to perform validation, display this message
                logger.warning(
                    "Cannot perform validation, not supported for LocalJob"
                )

        # handle case when model is in a file
        if not isinstance(training_plan, TrainingPlan):
            raise TypeError(
                "'training_plan' should be a TrainingPlan instance."
            )
        self._training_plan = training_plan
        self._training_plan.post_init(
            model_args=self._model_args, training_args=self._training_args
        )

    @property
    def training_plan(self):
        return self._training_plan

    @property
    def model(self):
        return self._training_plan.model

    @property
    def training_args(self):
        return self._localjob_training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self._localjob_training_args = training_args

    def start_training(self):
        """Sends training task to nodes and waits for the responses"""

        is_failed = False
        error_message = ""

        # Run the training routine
        if not is_failed:
            results = {}
            try:
                base_manager = self._training_plan.training_data(
                    self.dataset_path
                )
                data_manager = base_manager.build(
                    self._training_plan.data_loader_type()
                )
                train_loader, test_loader = data_manager.split(test_ratio=0)
                self._training_plan.set_data_loaders(train_loader, test_loader)
                self._training_plan.training_routine()
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model in job : " + str(e)

        if not is_failed:
            try:
                # TODO : should test status code but not yet returned
                # by upload_file
                filename = (
                    environ["TMP_DIR"]
                    + "/local_params_"
                    + str(uuid.uuid4())
                    + ".json"
                )
                self._training_plan.save(filename, results)
            except Exception as e:
                is_failed = True
                error_message = "Cannot write results: " + str(e)

        if error_message != "":
            logger.error(error_message)
