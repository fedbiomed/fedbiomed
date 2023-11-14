# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implements the message exchanges from researcher to nodes
"""

import json
import os
import uuid
import tempfile
import threading
from typing import Any, Dict, Callable, Union, List, Optional

import tabulate
from python_minifier import minify

from fedbiomed.common.constants import MessageType
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ResearcherMessages, SearchRequest, ErrorMessage, Message
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.utils import import_class_object_from_file

from fedbiomed.transport.server import GrpcServer
from fedbiomed.transport.node_agent import NodeAgent, NodeActiveStatus

from fedbiomed.researcher.environ import environ

from ._policies import RequestPolicy, PolicyController, DiscardOnTimeout
from ._status import RequestStatus, PolicyStatus

# timeout in seconds for checking disconnection and node status changes
REQUEST_STATUS_CHECK_TIMEOUT = 0.5


class MessagesByNode(dict):
    """Type to defined messages by node"""
    pass


class Request:

    def __init__(
        self,
        message: Message,
        node: NodeAgent,
        request_id: Optional[str] = None,
        sem_pending: threading.Semaphore = None
    ) -> None:
        """Single request for node

        Args:
            message: Message to send to the node
            node: Node agent
            request_id: unique ID of request
            sem_pending: semaphore for signaling new pending reply
        """
        self.request_id = request_id if request_id else str(uuid.uuid4())
        self.node = node
        self.message = message

        self._sem_pending = sem_pending

        self.reply = None
        self.error = None
        self.status = None

    def has_finished(self) -> bool:
        """Queries if the request has finished.

        Also tracks if node has disconnected.

        Args:
            True if a reply was received from node
        """

        if self.node.status == NodeActiveStatus.DISCONNECTED:
            self.status = RequestStatus.DISCONNECT
            return True  # Don't expect any reply

        return True if self.reply or self.error else False

    def send(self) -> None:
        """Sends the request"""
        self.message.request_id = self.request_id
        self.node.send(self.message, self.on_reply)
        self.status = RequestStatus.NO_REPLY_YET

    def flush(self, stopped: bool) -> None:
        """Flushes the reply that has been processed

        Args:
            stopped: True if the request was stopped before completion
        """
        self.node.flush(self.request_id, stopped)

    def on_reply(self, reply: dict) -> None:
        """Callback for node agent to execute once it replies.

        Args:
            reply: reply message received from node
        """

        if isinstance(reply, ErrorMessage):
            self.error = reply
            self.status = RequestStatus.ERROR
        else:
            self.reply = reply
            self.status = RequestStatus.SUCCESS

        self._sem_pending.release()


class FederatedRequest:
    """Dispatches federated requests

    This class has been design to be send a request and wait until a
    response is received
    """
    def __init__(
        self,
        message: Union[Message, MessagesByNode],
        nodes: List[NodeAgent],
        policy: Optional[List[RequestPolicy]] = None
    ):
        """Constructor of the class.

        Args:
            message: either a common `Message` to send to nodes or a dict of distinct message per node
                indexed by the node ID
            nodes: list of nodes that are sent the message
            policy: list of policies for controlling the handling of the request
        """

        self.message = message
        self.nodes = nodes
        self.requests = []
        self.request_id = str(uuid.uuid4())
        self.nodes_status = {}

        self._pending_replies = threading.Semaphore(value=0)

        # Set-up policies
        self.policy = PolicyController(policy)

        # Set up single requests
        if isinstance(self.message, Message):
            for node in self.nodes:
                self.requests.append(
                    Request(self.message, node, self.request_id, self._pending_replies)
                )

        # Different message for each node
        elif isinstance(self.message, MessagesByNode):
            for node in self.nodes:
                if m := self.message.get(node.id):
                    self.requests.append(
                        Request(m, node, self.request_id, self._pending_replies)
                    )
                else:
                    logger.warning(f"Node {node.id} is unknown. Send message to others, ignore this one.")

    def __enter__(self):
        """Context manager entry method

        Returns:
            The class object
        """

        # Sends the message
        self.send()

        # Blocking function that waits for the relies
        self.wait()

        return self

    def __exit__(self, type, value, traceback):
        """Context manager exit method

        Args:
            type: ignored
            value: ignored
            traceback: ignored
        """

        # Clear the replies that are processed
        has_stopped = self.policy.has_stopped_any()
        for req in self.requests:
            req.flush(stopped=has_stopped)

    def replies(self) -> Dict[str, Message]:
        """Returns replies of each request

        Returns:
            A dict of replies `Message` received for this request, indexed by node ID 
        """

        return {req.node.id: req.reply for req in self.requests if req.reply}

    def errors(self) -> Dict[str, ErrorMessage]:
        """Returns errors of each request

        Returns:
            A dict of error `Message` received for this request, indexed by node ID         
        """

        return {req.node.id: req.error for req in self.requests if req.error}

    def disconnected_requests(self) -> List[Message]:
        """Returns the requests to disconnected nodes

        Returns:
            A list of request `Message` sent to disconnected nodes
        """
        return [req for req in self.requests if req.status == RequestStatus.DISCONNECT]

    def send(self) -> None:
        """Sends federated request"""
        for req in self.requests:
            req.send()

    def wait(self) -> None:
        """Waits for the replies of the messages that are sent"""

        while self.policy.continue_all(self.requests) == PolicyStatus.CONTINUE:
            self._pending_replies.acquire(timeout=REQUEST_STATUS_CHECK_TIMEOUT)


class Requests(metaclass=SingletonMeta):
    """
    Manages communication between researcher and nodes.
    """

    def __init__(self):
        """Constructor of the class
        """
        self._monitor_message_callback = None

        # Creates grpc server and starts it
        self._grpc_server = GrpcServer(
            host=environ["SERVER_HOST"],
            port=environ["SERVER_PORT"],
            on_message=self.on_message
        )
        self.start_messaging()


    def start_messaging(self) -> None:
        """Start communications endpoint
        """
        self._grpc_server.start()

    def on_message(self, msg: Union[Dict[str, Any], Message], type_: MessageType) -> None:
        """Handles arbitrary messages received from the remote agents

        This callback is only used for feedback messages from nodes (logs, experiment
        monitor), not for node replies to requests.

        Args:
            msg: de-serialized msg
            type_: Reply type one of reply, log, scalar
        """

        if type_ == MessageType.LOG:
            # forward the treatment to node_log_handling() (same thread)
            self.print_node_log_message(msg.get_dict())

        elif type_ == MessageType.SCALAR:
            if self._monitor_message_callback is not None:
                # Pass message to Monitor's on message handler
                self._monitor_message_callback(msg.get_dict())
        else:
            logger.error(f"Undefined message type received  {type_} - IGNORING")

    @staticmethod
    def print_node_log_message(log: Dict[str, Any]) -> None:
        """Prints logger messages coming from the node

        It is run on the communication process and must be as quick as possible:
        - all logs (coming from the nodes) are forwarded to the researcher logger
        (immediate display on console/file/whatever)

        Args:
            log: log message and its metadata
        """

        # log contains the original message sent by the node
        # FIXME: we should use `fedbiomed.common.json.deserialize` method
        # instead of the json method when extracting json message
        original_msg = json.loads(log["msg"])

        # Loging fancy feedback for training
        logger.info("\033[1m{}\033[0m\n"
                    "\t\t\t\t\t\033[1m NODE\033[0m {}\n"
                    "\t\t\t\t\t\033[1m MESSAGE:\033[0m {}\033[0m\n"
                    "{}".format(log["level"],
                                log["node_id"],
                                original_msg["message"],
                                5 * "-------------"))

    def ping_nodes(self) -> list:
        """ Pings online nodes

        Returns:
            List of ID of up and running nodes
        """
        ping = ResearcherMessages.format_outgoing_message({
            'researcher_id': environ["ID"],
            'command': "ping"}
        )
        with self.send(ping) as federated_req:
            nodes_online = [node_id for node_id, reply in federated_req.replies().items()]

        return nodes_online

    def send(
            self,
            message: Union[Message, MessagesByNode],
            nodes: Optional[List[str]] = None,
            policies: List[RequestPolicy] = None
    ) -> FederatedRequest:
        """Sends federated request to given nodes with given message

        Args:
            message: either a common `Message` to send to nodes or a dict of distinct message per node
                indexed by the node ID
            nodes: list of nodes that are sent the message. If None, send the message to all known active nodes.
            policy: list of policies for controlling the handling of the request, or None

        Returns:
            The object for handling the communications for this request
        """

        if nodes is not None:
            nodes = [self._grpc_server.get_node(node) for node in nodes]
        else:
            nodes = self._grpc_server.get_all_nodes()

        return FederatedRequest(message, nodes, policies)

    def search(self, tags: tuple, nodes: Optional[list] = None) -> dict:
        """Searches available data by tags

        Args:
            tags: Tuple containing tags associated to the data researcher is looking for.
            nodes: optionally filter nodes with this list. Default is None, no filtering, consider all nodes

        Returns:
            A dict with node_id as keys, and list of dicts describing available data as values
        """

        message = SearchRequest(
            tags=tags,
            researcher_id=environ['RESEARCHER_ID'],
            command='search'
        )

        data_found = {}
        with self.send(message, nodes, policies=[DiscardOnTimeout(5)]) as federated_req:

            for node_id, reply in federated_req.replies().items():
                if reply.databases:
                    data_found[node_id] = reply.databases
                    logger.info('Node selected for training -> {}'.format(reply.node_id))

            for node_id, error in federated_req.errors().items():
                logger.warning(f"Node {node_id} has returned error from search request {error.extra_msg}")


            if not data_found:
                logger.info("No available dataset has found in nodes with tags: {}".format(tags))

        return data_found


    def list(self, nodes: Optional[list] = None, verbose: bool = False) -> dict:
        """Lists available data in each node

        Args:
            nodes: optionally filter nodes with this list. Default is None, no filtering, consider all nodes
            verbose: If it is true it prints datasets in readable format

        Returns:
            A dict with node_id as keys, and list of dicts describing available data as values
        """

        message = ResearcherMessages.format_outgoing_message(
            {"researcher_id": environ['RESEARCHER_ID'],
             "command": "list"}
        )

        data_found = {}
        with self.send(message, nodes, policies=[DiscardOnTimeout(5)]) as federated_req:
            for node_id, reply in federated_req.replies().items():
                data_found[node_id] = reply.databases

        if verbose:
            for node in data_found:
                if len(data_found[node]) > 0:
                    rows = [row.values() for row in data_found[node]]
                    headers = data_found[node][0].keys()
                    info = '\n Node: {} | Number of Datasets: {} \n'.format(node, len(data_found[node]))
                    logger.info(info + tabulate.tabulate(rows, headers, tablefmt="grid") + '\n')
                else:
                    logger.info('\n Node: {} | Number of Datasets: {}'.format(node, len(data_found[node])) +
                                " No data has been set up for this node.")

        return data_found

    def training_plan_approve(
            self,
            training_plan: BaseTrainingPlan,
            description: str = "no description provided",
            nodes: list = []
    ) -> dict:
        """Send a training plan and a ApprovalRequest message to node(s).

        If a list of node id(s) is provided, the message will be individually sent
        to all nodes of the list.
        If the node id(s) list is None (default), the message is broadcast to all nodes.

        Args:
            training_plan: the training plan class to send to the nodes for approval.
            description: Description of training plan approval request
            nodes: list of nodes (specified by their UUID)

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the training plan has been correctly downloaded on the node side.
            Warning: status does not mean that the training plan is approved, only that it has been added
            to the "approval queue" on the node side.
        """

        training_plan_instance = training_plan()
        training_plan_module = 'my_model_' + str(uuid.uuid4())
        with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as tmp_dir:
            training_plan_file = os.path.join(tmp_dir, training_plan_module + '.py')
            try:
                training_plan_instance.save_code(training_plan_file)
            except Exception as e:
                logger.error(f"Cannot save the training plan to a local tmp dir : {e}")
                return {}

            try:
                _, training_plan_instance = import_class_object_from_file(
                    training_plan_file, training_plan.__name__)
                tp_source = training_plan_instance.source()
            except Exception as e:
                logger.error(f"Cannot instantiate the training plan: {e}")
                return {}

        try:
            minify(tp_source,
                   remove_annotations=False,
                   combine_imports=False,
                   remove_pass=False,
                   hoist_literals=False,
                   remove_object_base=True,
                   rename_locals=False)
        except Exception as e:
            # minify does not provide any specific exception
            logger.error(f"This file is not a python file ({e})")
            return {}

        # send message to node(s)
        message = ResearcherMessages.format_outgoing_message({
            'researcher_id': environ['RESEARCHER_ID'],
            'description': str(description),
            'training_plan': tp_source,
            'command': 'approval'})

        with self.send(message, nodes, policies=[DiscardOnTimeout(5)]) as federated_req:
            errors = federated_req.errors()
            replies = federated_req.replies()

            # TODO: Loop over errors and replies
            for node_id in nodes:
                if node_id in errors:
                    logger.info(f"Node ({node_id}) has returned error {errors[node_id].errnum}, {errors[node_id].extra_msg}")

                if node_id not in replies:
                    logger.info(f"Node ({node_id}) has not replied")

            return replies

    def add_monitor_callback(self, callback: Callable[[Dict], None]):
        """ Adds callback function for monitor messages

        Args:
            callback: Callback function for handling monitor messages that come due 'general/monitoring' channel
        """

        self._monitor_message_callback = callback

    def remove_monitor_callback(self):
        """ Removes callback function for Monitor class. """

        self._monitor_message_callback = None
