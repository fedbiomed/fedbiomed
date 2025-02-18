# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Core code of the node component.
"""

import os

from typing import Callable, Optional, Union

from fedbiomed.common.constants import ErrorNumbers, CONFIG_FOLDER_NAME
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    AdditiveSSSetupRequest,
    ApprovalRequest,
    ErrorMessage,
    Message,
    OverlayMessage,
    PingReply,
    PingRequest,
    SearchReply,
    SearchRequest,
    ListRequest,
    ListReply,
    SecaggDeleteReply,
    SecaggDeleteRequest,
    SecaggReply,
    SecaggRequest,
    TrainingPlanStatusRequest,
    TrainRequest,
)
from fedbiomed.common.synchro import EventWaitExchange
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.config import NodeConfig
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.requests import NodeToNodeRouter
from fedbiomed.node.round import Round
from fedbiomed.node.secagg import SecaggSetup
from fedbiomed.node.secagg_manager import SecaggManager
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager
from fedbiomed.transport.client import ResearcherCredentials
from fedbiomed.transport.controller import GrpcController


class Node:
    """Core code of the node component.

    Defines the behaviour of the node, while communicating
    with the researcher through the `Messaging`, parsing messages from the researcher,
    etiher treating them instantly or queuing them,
    executing tasks requested by researcher stored in the queue.
    """
    tp_security_manager: TrainingPlanSecurityManager
    dataset_manager: DatasetManager

    def __init__(
        self,
        config: NodeConfig,
        node_args: Union[dict, None] = None,
    ):
        """Constructor of the class.

        Attributes:
            config: Node configuration
            node_args: Command line arguments for node.
        """

        self._config = config
        self._node_id = self._config.get('default', 'id')
        self._tasks_queue = TasksQueue(
            os.path.join(self._config.root, 'var', f'queue_{self._node_id}'),
            str(os.path.join(self._config.root, 'var', 'tmp'))
        )

        self._grpc_client = GrpcController(
            node_id=self._node_id,
            researchers=[
                ResearcherCredentials(
                    port=self._config.get('researcher', 'port'),
                    host=self._config.get('researcher', 'ip'),
                    certificate=self._config.get('researcher', 'certificate', fallback=None)
                )
            ],
            on_message=self.on_message,
        )
        self._db_path = os.path.abspath(os.path.join(
            self._config.root, CONFIG_FOLDER_NAME, self._config.get('default', 'db'))
        )

        self._pending_requests = EventWaitExchange(remove_delivered=True)
        self._controller_data = EventWaitExchange(remove_delivered=False)
        self._n2n_router = NodeToNodeRouter(
            self._node_id,
            self._db_path,
            self._grpc_client,
            self._pending_requests,
            self._controller_data
        )


        self.dataset_manager = DatasetManager(db=self._db_path)
        self.tp_security_manager = TrainingPlanSecurityManager(
            db=self._db_path,
            node_id=self._node_id,
            hashing=self._config.get('security', 'hashing_algorithm'),
            tp_approval=self._config.getbool('security', 'training_plan_approval')
        )

        self.node_args = node_args

    @property
    def node_id(self):
        """Returns id of the node"""
        return self._node_id


    @property
    def config(self):
        """Return node config"""
        return self._config

    def add_task(self, task: dict):
        """Adds a task to the pending tasks queue.

        Args:
            task: A `Message` object describing a training task
        """
        self._tasks_queue.add(task)

    def on_message(self, msg: dict):
        """Handler to be used with `Messaging` class (ie the messager).

        Called when a  message arrives through the `Messaging`.
        It reads and triggers instructions received by node from Researcher,
        mainly:
        - ping requests,
        - train requests (then a new task will be added on node's task queue),
        - search requests (for searching data in node's database).

        Args:
            msg: Incoming message from Researcher.
        """
        message: Message
        try:
            message = Message.from_dict(msg)
        except FedbiomedError as e:
            logger.error(e)  # Message was not properly formatted
            resid = msg.get("researcher_id", "unknown_researcher_id")
            self.send_error(
                ErrorNumbers.FB301,
                extra_msg="Message was not properly formatted",
                researcher_id=resid,
            )
        else:
            no_print = [
                "aggregator_args",
                "optim_aux_var",
                "params",
                "training_plan",
                "overlay",
            ]
            msg_print = {
                key: value
                for key, value in message.get_dict().items()
                if key not in no_print
            }
            logger.debug("Message received: " + str(msg_print))

            match message.__name__:

                case (
                    TrainRequest.__name__
                    | SecaggRequest.__name__
                    | AdditiveSSSetupRequest.__name__
                ):
                    self.add_task(message)
                case SecaggDeleteRequest.__name__:
                    self._task_secagg_delete(message)
                case OverlayMessage.__name__:
                    self._n2n_router.submit(message)
                case SearchRequest.__name__:
                    databases = self.dataset_manager.search_by_tags(message.tags)
                    if len(databases) != 0:
                        databases = self.dataset_manager.obfuscate_private_information(
                            databases
                        )
                    self._grpc_client.send(
                        SearchReply(
                            request_id=message.request_id,
                            node_id=self._node_id,
                            researcher_id=message.researcher_id,
                            databases=databases,
                            count=len(databases),
                        )
                    )
                case ListRequest.__name__:
                    # Get list of all datasets
                    databases = self.dataset_manager.list_my_data(verbose=False)
                    databases = self.dataset_manager.obfuscate_private_information(databases)
                    self._grpc_client.send(
                        ListReply(
                            success=True,
                            request_id=message.request_id,
                            node_id=self._node_id,
                            researcher_id=message.researcher_id,
                            databases=databases,
                            count=len(databases),
                        )
                    )

                case PingRequest.__name__:
                    self._grpc_client.send(
                        PingReply(
                            request_id=message.request_id,
                            researcher_id=message.researcher_id,
                            node_id=self._node_id,
                        )
                    )
                case ApprovalRequest.__name__:
                    reply = (
                        self.tp_security_manager.reply_training_plan_approval_request(
                            message
                        )
                    )
                    self._grpc_client.send(reply)
                case TrainingPlanStatusRequest.__name__:
                    reply = self.tp_security_manager.reply_training_plan_status_request(
                        message
                    )
                    self._grpc_client.send(reply)
                case _:
                    resid = msg.get("researcher_id", "unknown_researcher_id")
                    self.send_error(
                        ErrorNumbers.FB301,
                        extra_msg="This request handler is not implemented "
                        f"{message.__class__.__name__} is not implemented",
                        researcher_id=resid,
                    )

    def _task_secagg_delete(self, msg: SecaggDeleteRequest) -> None:
        """Parse a given secagg delete task message and execute secagg delete task.

        Args:
            msg: `SecaggDeleteRequest` message object to parse
        """

        secagg_id = msg.secagg_id

        reply = {
            "node_id": self._node_id,
            "researcher_id": msg.researcher_id,
            "success": True,
            "secagg_id": secagg_id,
            "request_id": msg.request_id,
        }

        secagg_manager = SecaggManager(
            db=self._db_path,
            element=msg.get_param("element")
        )()

        status = secagg_manager.remove(
            secagg_id=secagg_id, experiment_id=msg.get_param("experiment_id")
        )

        if not status:
            message = (
                f"{ErrorNumbers.FB321.value}: no such secagg context "
                f"element in node database for node_id={self._node_id} "
                f"secagg_id={secagg_id}"
            )
            return self.send_error(
                extra_msg=message,
                researcher_id=msg.researcher_id,
                request_id=msg.request_id,
            )

        return self._grpc_client.send(
            SecaggDeleteReply(**{**reply, "msg": "Secagg context is deleted."})
        )

    def _task_secagg(self, request: SecaggRequest) -> None:
        """Parse a given secagg setup task message and execute secagg task.

        Args:
            request: `SecaggRequest` message object to parse
        """
        setup_arguments = request.get_dict()
        setup_arguments.pop('protocol_version')
        setup_arguments.pop('request_id')

        # Needed when using node to node communications
        setup_arguments.update({
            'db': self._db_path,
            'node_id': self._node_id,
            'n2n_router': self._n2n_router,
            'grpc_client': self._grpc_client,
            'pending_requests': self._pending_requests,
            'controller_data': self._controller_data,
        })

        try:
            secagg = SecaggSetup(**setup_arguments)()
            reply: SecaggReply = secagg.setup()
        except Exception as error_message:
            logger.error(error_message)
            return self.send_error(
                request_id=request.request_id,
                researcher_id=request.researcher_id,
                extra_msg=str(error_message),
            )

        reply.request_id = request.request_id
        return self._grpc_client.send(reply)

    def parser_task_train(self, msg: TrainRequest) -> Union[Round, None]:
        """Parses a given training task message to create a round instance

        Args:
            msg: `TrainRequest` message object to parse

        Returns:
            a `Round` object for the training to perform, or None if no training
        """
        round_ = None
        hist_monitor = HistoryMonitor(
            node_id=self._node_id,
            experiment_id=msg.experiment_id,
            researcher_id=msg.researcher_id,
            send=self._grpc_client.send,
        )

        dataset_id = msg.get_param("dataset_id")
        data = self.dataset_manager.get_by_id(dataset_id)

        if data is None:
            return self.send_error(
                extra_msg="Did not found proper data in local datasets "
                f'on node={self._node_id}',
                request_id=msg.request_id,
                researcher_id=msg.researcher_id,
                errnum=ErrorNumbers.FB313,
            )

        dlp_and_loading_block_metadata = None
        if "dlp_id" in data:
            dlp_and_loading_block_metadata = self.dataset_manager.get_dlp_by_id(
                data["dlp_id"]
            )

        round_ = Round(
            root_dir=self._config.root,
            db=self._db_path,
            node_id=self._node_id,
            training_plan=msg.get_param("training_plan"),
            training_plan_class=msg.get_param("training_plan_class"),
            model_kwargs=msg.get_param("model_args") or {},
            training_kwargs=msg.get_param("training_args") or {},
            training=msg.get_param("training") or False,
            dataset=data,
            params=msg.get_param("params"),
            experiment_id=msg.get_param("experiment_id"),
            researcher_id=msg.get_param("researcher_id"),
            history_monitor=hist_monitor,
            aggregator_args=msg.get_param("aggregator_args") or None,
            node_args=self.node_args,
            tp_security_manager=self.tp_security_manager,
            round_number=msg.get_param("round"),
            dlp_and_loading_block_metadata=dlp_and_loading_block_metadata,
            aux_vars=msg.get_param('optim_aux_var'),
        )

        # the round raises an error if it cannot initialize
        err_msg = round_.initialize_arguments(msg.get_param("state_id"))
        if err_msg is not None:
            self._grpc_client.send(
                ErrorMessage(
                    node_id=self._node_id,
                    errnum=ErrorNumbers.FB300,
                    researcher_id=msg.researcher_id,
                    extra_msg="Could not initialize arguments",
                )
            )

        return round_

    def task_manager(self):
        """Manages training tasks in the queue."""

        while True:

            item: TrainRequest = self._tasks_queue.get()
            # don't want to treat again in case of failure
            self._tasks_queue.task_done()

            logger.info(
                f"[TASKS QUEUE] Task received by task manager: "
                f"Researcher: {item.researcher_id} "
                f"Experiment: {item.experiment_id}"
            )
            try:

                match item.__name__:
                    case TrainRequest.__name__:
                        round_ = self.parser_task_train(item)
                        # once task is out of queue, initiate training rounds
                        if round_ is not None:
                            msg = round_.run_model_training(
                                tp_approval=self._config.getbool('security', 'training_plan_approval'),
                                secagg_insecure_validation=self._config.getbool('security',
                                    "secagg_insecure_validation"),
                                secagg_active=self._config.getbool("security", "secure_aggregation"),
                                force_secagg=self._config.getbool(
                                    "security", "force_secure_aggregation"),
                                secagg_arguments=item.get_param("secagg_arguments"),
                            )
                            msg.request_id = item.request_id
                            self._grpc_client.send(msg)
                            del round_

                    case SecaggRequest.__name__ | AdditiveSSSetupRequest.__name__:
                        self._task_secagg(item)
                    case _:
                        errmess = f"{ErrorNumbers.FB319.value}: Undefined request message"
                        self.send_error(errnum=ErrorNumbers.FB319, extra_msg=errmess)

            # TODO: Test exception
            except Exception as e:
                self.send_error(
                    request_id=item.request_id,
                    researcher_id=item.researcher_id,
                    errnum=ErrorNumbers.FB300,
                    extra_msg="Round error: " + str(e),
            )

    def start_protocol(self) -> None:
        """Start the node to node router thread, for handling node to node message"""
        self._n2n_router.start()

    def start_messaging(self, on_finish: Optional[Callable] = None):
        """Calls the start method of messaging class.

        Args:
            on_finish: Called when the tasks for handling all known researchers have finished.
                Callable has no argument.
        """
        self._grpc_client.start(on_finish)

    def is_connected(self) -> bool:
        """Checks if node is ready for communication with researcher

        Returns:
            True if node is ready, False if node is not ready
        """
        return self._grpc_client.is_connected()

    def send_error(
        self,
        errnum: ErrorNumbers = ErrorNumbers.FB300,
        extra_msg: str = "",
        researcher_id: str = "<unknown>",
        broadcast: bool = False,
        request_id: str = None,
    ):
        """Sends an error message.

        Args:
            errnum: Code of the error.
            extra_msg: Additional human readable error message.
            researcher_id: Destination researcher.
            broadcast: Broadcast the message all available researchers
                regardless of specific researcher.
            request_id: Optional request i to reply as error to a request.
        """
        try:
            logger.error(extra_msg)
            self._grpc_client.send(
                ErrorMessage(
                    request_id=request_id,
                    errnum=errnum.name,
                    node_id=self._node_id,
                    extra_msg=extra_msg,
                    researcher_id=researcher_id,
                ),
                broadcast=broadcast,
            )
        except Exception as e:
            logger.error(f"{ErrorNumbers.FB601.value}: Cannot send error message: {e}")
