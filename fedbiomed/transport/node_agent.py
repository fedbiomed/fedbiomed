"""
This module contains
"""

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Callable, Awaitable
from datetime import datetime
import copy
import time
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional

import grpc

from fedbiomed.common.logger import logger
from fedbiomed.common.message import Message, OverlayMessage

# timeout in seconds for server to wait for a new task request from node before assuming node is disconnected
GRPC_SERVER_TASK_WAIT_TIMEOUT = 10


class NodeActiveStatus(Enum):
    """Node active status types

    Attributes:
        WAITING: Corresponds status where researcher server waits another GetTask request after
            the previous one is completed.
        ACTIVE: Listening for the task with open RPC call
        DISCONNECTED: No GetTask RPC call running from the node
    """

    WAITING = 1
    ACTIVE = 2
    DISCONNECTED = 3


class Replies(dict):
    pass


class NodeAgent(ABC):
    """Agent is class respecting a remote instance

    Node agent represents a remote node to send and receive messages
    """
    def __init__(self, id_: str) -> str:
        """Constructor"""
        self._id = id_
        self._replies = Replies()
        self._stopped_request_ids = []

    @property
    def id(self) -> str:
        """Return node id.

        Returns:
            node id
        """
        return self._id

    @abstractmethod
    @property
    def status(self):
        """Returns status of the agent"""

    @abstractmethod
    def send(self, message: Message, on_reply: Callable):
        """Sends message to the node"""

    def _register_reply(self, message: Message, on_reply: Callable) -> None:
        """Registers replies and its callback"""

        self._replies.update({
            message.request_id: {'callback': on_reply, 'reply': None}
        })


    def _handle_reply(self, message):
        """Once agent receives a replies from it remote twin"""

        if message.request_id in self._replies:
            if self._replies[message.request_id]['reply'] is None:
                self._replies[message.request_id]['reply'] = message
                self._replies[message.request_id]['callback'](message)
            else:
                # Handle case of multiple replies avoid conflict with consumption of reply.
                logger.warning(
                    f"Received multiple replies for request {message.request_id}. "
                    "Keep first reply, ignore subsequent replies")
        else:
            if message.request_id in self._stopped_request_ids:
                logger.warning(
                    "Received a reply from a federated request that has been "
                    f"stopped: {message.request_id}.")
                self._stopped_request_ids.remove(message.request_id)
            else:
                logger.warning(
                    f"Received a reply from an unexpected request: {message.request_id}")



    def flush(self, request_id: str, stopped: bool = False):
        """Flushes processed reply

        Args:
            request_id: request ID for which the replies should be flushed
            stopped: the request was stopped during processing
        """

        if stopped and self._replies[request_id]['reply'] is None:
            self._stopped_request_ids.append(request_id)

        self._replies.pop(request_id, None)


class NodeAgentAsync(NodeAgent):
    """Async implementation of node agent"""

    def __init__(
        self,
        id: str,
        loop: asyncio.AbstractEventLoop,
        on_forward: Optional[Awaitable[None]],
   ) -> None:
        """Represent the client that connects to gRPC server

        Args:
            id: node unique ID
            loop: event loop
            on_forward: Coroutine for handling overlay messages to forward unchanged to a node
        """
        super().__init__(id_)

        self._on_forward = on_forward
        self._last_request: Optional[datetime] = None
        # Node should be active when it is first instantiated
        self._status: NodeActiveStatus = NodeActiveStatus.ACTIVE

        self._queue = asyncio.Queue()
        self._loop = loop
        self._status_task: Optional[asyncio.Task] = None

        # protect read/write operations on self._status + self._status_task)
        self._status_lock = asyncio.Lock()
        self._replies_lock = asyncio.Lock()
        self._stopped_request_ids_lock = asyncio.Lock()

    async def status_async(self) -> NodeActiveStatus:
        """Getter for node status.

        Returns:
            node status
        """
        async with self._status_lock:
            # (deep)copy is not needed as long as this remains a simple value ...
            return self._status

    async def on_reply(self, message: Dict) -> None:
        """Callback to execute each time new reply received from the node"""

        message = Message.from_dict(message)

        # Handle overlay messages to relay to a node
        if isinstance(message, OverlayMessage):
            await self._on_forward(message)
            return

        # Common case where callback is executed
        self._handle_reply(message)

    async def send_async(
        self,
        message: Message,
        on_reply: Optional[Callable] = None,
        retry_count: int = 0,
        first_send_time: Optional[float] = None,
    ) -> None:
        """Async function send message to researcher.

        Args:
            message: Message to send to the researcher
            on_reply: optional callback to execute when receiving message reply
            retry_count: number of retries already done for this message
            first_send_time: time of first send attempt for this message
        """

        if self._status == NodeActiveStatus.DISCONNECTED:
            logger.info(f"Node {self._id} is disconnected. Discard message.")
            return

        if self._status == NodeActiveStatus.WAITING:
            logger.info(f"Node {self._id} is in WAITING status. Server is "
                        "waiting for receiving a request from "
                        "this node to convert it as ACTIVE. Node will be updated "
                        "as DISCONNECTED soon if no request received.")

        self._register_reply(message, on_reply)

        if first_send_time is None:
            first_send_time = time.time()

        await self._queue.put((message, retry_count, first_send_time))


    async def set_active(self) -> None:
        """Updates node status as active"""

        async with self._status_lock:

            # Inform user that node is online again
            if self._status == NodeActiveStatus.DISCONNECTED:
                logger.info(f"Node {self._id} is back online!")

            self._status = NodeActiveStatus.ACTIVE

            # Cancel status task if there is any running
            if self._status_task:
                self._status_task.cancel()
                self._status_task = None

    async def change_node_status_after_task(self) -> None:
        """Coroutine to execute each time RPC call is completed"""
        async with self._status_lock:
            self._status = NodeActiveStatus.WAITING

            if self._status_task is None:
                self._status_task = asyncio.create_task(
                    self._change_node_status_disconnected()
                )

    async def _change_node_status_disconnected(self) -> None:
        """Task coroutine to change node status as `DISCONNECTED` after a delay

        Node becomes DISCONNECTED if it doesn't become ACTIVE in GRPC_SERVER_TASK_WAIT_TIMEOUT seconds,
        which cancels this task
        """

        # Sleep at least GRPC_SERVER_TASK_WAIT_TIMEOUT seconds in WAITING
        await asyncio.sleep(GRPC_SERVER_TASK_WAIT_TIMEOUT)

        # If the status still WAITING set status to DISCONNECTED
        async with self._status_lock:
            if self._status == NodeActiveStatus.WAITING:
                self._status = NodeActiveStatus.DISCONNECTED
                logger.warning(
                    f"Node {self._id} is disconnected. Request/task that are created "
                    "for this node will be flushed"
                )
                # TODO: empty the queue when becoming disconnected ?


class NodeAgentServer(NodeAgentAsync):
    """Node agent used in server"""

    @property
    def status(self) -> NodeActiveStatus:
        """Getter for node status.

        Returns:
            node status
        """
        future = asyncio.run_coroutine_threadsafe(self.status_async(), self._loop)
        return future.result()

    def flush(self, request_id: str, stopped: bool = False) -> None:
        """Flush processed replies

        Args:
            request_id: request ID for which the replies should be flushed
            stopped: the request was stopped during processing
        """
        asyncio.run_coroutine_threadsafe(super().flush(request_id, stopped), self._loop)

    def send(self, message: Message, on_reply: Optional[Callable] = None) -> None:
        """Send message to researcher.

        Args:
            message: Message to send to the researcher
        """
        asyncio.run_coroutine_threadsafe(
            self.send_async(message=message, on_reply=on_reply), self._loop
        )

    def task_done(self) -> None:
        """Acknowledge completion of a de-queued task
        """
        self._queue.task_done()

    def get_task(self) -> Awaitable[Message]:
        """Get tasks assigned by the main thread

        !!! note "Returns coroutine"
            This function return an asyncio coroutine. Please use `await` while calling.

        Returns:
            A coroutine to await for retrieving a list of: a task ; a number of send retries already done ;
                the time of first sending attempt in seconds since epoch
        """
        return self._queue.get()




class NodeAgentN2N(NodeAgent):
    """Node agent on the client side for N2N message"""

    def __init__(self, grpc_controller: GrpcController, **kwargs):
        super().__init__(**kwargs)
        self._grpc_client = grpc_controller

    def send(self, message, on_reply):
        """Sends and registers reply callback"""

        self._register_reply(message, on_reply)

        self._grpc_client.send(message, id)



class AgentStore(ABC):
    """Agent store to keep node agents

    Attributes:
        _agents: A dict containing all node agents saved

    """

    def __init__(self):
        """Constructs agent store"""
        self._agents: NodeAgent = {}

    @abstractmethod
    def _create_agent(self, agent_id: str) -> NodeAgent:
        """Creates agent"""

    def retrieve(
            self,
            node_id: str
    ) -> NodeAgent:
        """Retrieves a node agent for a given node ID.

        Depending if this node is already known to the store this method gets existing agent or
        registers a new agent.

        Args:
            node_id: ID of receiving node

        Return:
            The node agent to manage tasks that are assigned to it.
        """
        # Lock during all sequence to ensure atomicity
        node = self._agents.get(node_id)

        if not node:
            node = self._create_agent(node_id)
            self._agents.update({node_id: node})

        return node

    def get_all(self) -> Dict[str, NodeAgent]:
        """Returns all node agents regardless of their status (ACTIVE, DISCONNECTED, ...).

        Returns:
            Dictionary of node agent objects, by node ID
        """

        return copy.copy(self._agents)

    def get(
            self,
            node_id: str
    ) -> Optional[NodeAgent]:
        """Gets node agent by given node id

        Args:
            node_id: Id of the node, or None if no agent exists for this node ID
        """
        return self._agents.get(node_id)


class AgentStoreClientSide(AgentStore):
    """Agent store to be used on the client side

    This class is manage to node agents on the client side to handle
    node to node messages
    """
    def __init__(self, grpc_controller: 'GrpcController', **kwargs):
        """Constructs client side agent store"""

        super().__init__(**kwargs)
        self._grpc_controller = grpc_controller


    def _create_agent(self, agent_id: str) -> NodeAgent:
        """Method to create an agent"""

        return NodeAgentN2N(
            grpc_controller=self._grpc_controller,
            id=agent_id)



class AgentStoreServer(AgentStore):
    """Stores node agents"""

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 on_forward: Awaitable[None] | None = None) -> None:
        """Constructor of the agent store

        Args:
            loop: asyncio event loop that research server runs. Agent store should use
                same event loop for async operations
            on_forward: Coroutine for handling overlay messages to forward unchanged to a node
        """
        self._loop = loop
        self._on_forward = on_forward

    def _create_agent(self, node_id: str):
        """Overwrites create agent to create NodeAgent for server side"""

        return NodeAgent(id=node_id, loop=self._loop, on_forward=self._on_forward)

