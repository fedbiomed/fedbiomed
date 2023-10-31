# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implements the message exchanges from researcher to nodes
"""

import json
import os
import tabulate
import uuid
import tempfile
from time import sleep
from typing import Any, Dict, Callable, Union, List, Optional, Tuple

from python_minifier import minify

from fedbiomed.common.constants import MessageType
from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ResearcherMessages, SearchRequest, SearchReply, ErrorMessage, Message
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.utils import import_class_object_from_file

from fedbiomed.transport.server import GrpcServer
from fedbiomed.transport.node_agent import NodeAgent, NodeActiveStatus

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses

from ._strategies import ContinueOnDisconnect, \
    ContinueOnError, \
    StopOnAnyDisconnect, \
    StopOnAnyError, \
    RequestStrategy, \
    StrategyController
    

class FederatedRequest: 
    """Dispatches federated requests 
    
    This class has been design to be send a request and wait until a
    response is received 
    """
    def __init__(
        self, 
        nodes: List[NodeAgent], 
        message: Union[Message, Dict[str, Message]], 
        strategy: Optional[RequestStrategy] = [ContinueOnDisconnect(), ContinueOnError()]
    ):

        self.request_id = str(uuid.uuid4())
        self.replies = {}
        self.errors = {}
        self.nodes = nodes
        self.message = message
        self.strategy = StrategyController(strategy)
        self.strategy.update(self.nodes, self.replies, self.errors) 

    def __enter__(self):

        # Sends the message 
        self.send()

        # Blocking function that waits for the relies
        self.wait()

        return self.replies, self.errors
    
    def __exit__(self, type, value, traceback):
        """Clear the replies that are processed"""
        for node in self.nodes:
            node.flush(self.request_id)

    def send(self):
        """Sends the message"""
        if isinstance(self.message, Message):
            self.message.request_id = self.request_id
            for node in self.nodes: 
                node.send(self.message, self.on_reply)
        elif isinstance(self.message, dict):
            for node in self.nodes: 
                m = self.message.get(node.id)
                m.request_id = self.request_id
                node.send(m, self.on_reply)
        else:
            raise TypeError(
                "The message should be an instance of Message or dictionary of message where " 
                "each key represents nodes."
            )

    def wait(self):
        """Waits for the replies of the messages that are sent"""

        while self.strategy.continue_():
            sleep(0.2)
            self.strategy.update(self.nodes, self.replies, self.errors)  
             
    def on_reply(self, message: Message, node_id: str):
        """Callback to execute once a reply is received"""

        if isinstance(message, ErrorMessage):
            self.errors.update({node_id: message})
        else:
            self.replies.update({node_id: message})

    

class Requests(metaclass=SingletonMeta):
    """
    Represents the requests addressed from Researcher to nodes. It creates a task queue storing reply to each
    incoming message. Starts a message queue and reconfigures  message to be sent into a `Messaging` object.
    """

    def __init__(self):
        """
        Constructor of the class

        Args:
            mess: message to be sent by default.
        """
        # Need to ensure unique per researcher instance message queue to avoid conflicts
        # in case several instances of researcher (with same researcher_id ?) are active,
        # eg: a notebook not quitted and launching a script
        self.queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'] + '_' + str(uuid.uuid4()), environ['TMP_DIR'])

        # defines the sequence used for ping protocol
        self._sequence = 0

        self._monitor_message_callback = None

        # Creates grpc server and starts it
        self._grpc_server = GrpcServer(
            host=environ["SERVER_HOST"],
            port=environ["SERVER_PORT"],
            on_message=self.on_message
        )
        self.start_messaging()


    def start_messaging(self):
        """Start communications endpoint
        """
        self._grpc_server.start()

    def on_message(self, msg: Union[Dict[str, Any], Message], type_: MessageType):
        """ Handler called by the [`ResearcherServer`][fedbiomed.transport.researcher_server] class,
        when a message is received on researcher side.

        It is run in the communication process and must be as quick as possible:
        - it deals with quick messages (eg: ping/pong)
        - it stores the replies of the nodes to the task queue, the message will be
        treated by the main (computing) thread.

        Args:
            msg: de-serialized msg
            type: Reply type one of reply, log, scalar
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
            List ids of up and running nodes
        """
        ping = ResearcherMessages.format_outgoing_message({
            'researcher_id': environ["ID"],
            'sequence': self._sequence,
            'command': "ping"}
        )
        with self.send(ping) as (replies, _):
            nodes_online = [node_id for node_id, reply in replies.items()]
        
        return nodes_online

    def send(
            self, 
            message: Union[Message, Dict[str, Message]], 
            nodes: Optional[List[str]] = None 
        ) -> FederatedRequest:
        """Sends federated request to given nodes with given message"""

        if nodes is not None:
            nodes = [self._grpc_server.get_node(node) for node in nodes]
        else:
            nodes = self._grpc_server.get_all_nodes()

        return FederatedRequest(nodes, message)

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
        with self.send(message, nodes) as (replies, errors):    
            for node_id, reply in replies.items():
                if reply.databases:
                    data_found[node_id] = reply.databases
                    logger.info('Node selected for training -> {}'.format(reply.node_id))

            for node_id, error in errors.items():
                logger.warning(f"Node {node_id} has returned error from search request {error.msg}")


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
        with self.send(message, nodes) as (replies, errors):
            for node_id, reply in replies.items():
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
            timeout: maximum waiting time for the answers

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

        with self.send(message, nodes) as (replies, errors):
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
