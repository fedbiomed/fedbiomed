from enum import Enum
from typing import Optional, Dict
from datetime import datetime
import copy

import asyncio
import grpc

from fedbiomed.common.message import Message
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.constants import ErrorNumbers


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


class NodeAgent:

    def __init__(
            self,
            id: str,
            loop,
    ) -> None:
        """Represent the client that connects to gRPC server"""
        self._id: str = id
        self._last_request: Optional[datetime] = None

        # Node should be active when it is first instantiated
        self._status: NodeActiveStatus = NodeActiveStatus.ACTIVE

        self._queue = asyncio.Queue()
        self._loop = loop
        self._status_task : Optional[asyncio.Task] = None
        self._status_lock = asyncio.Lock()


    @property
    async def status(self) -> NodeActiveStatus:
        async with self._status_lock:
            # deepcopy is not needed as long as this remains a simple value ...
            return self._status

    @property
    def id(self) -> str:
        return self._id

    async def active(self) -> None:
        """Updates node status as active"""

        async with self._status_lock:

            # Inform user that node is online again
            if self._status == NodeActiveStatus.DISCONNECTED:
                logger.info(f"Node {self._id} is back online!")

            self._status = NodeActiveStatus.ACTIVE
            self._last_request = datetime.now()

            # Cancel status task if there is any running
            if self._status_task:
                self._status_task.cancel()
                self._status_task = None

    async def send(self, message: Message) -> None:
        """Async function send message to researcher"""

        # TODO: this may happen, discard message ? put in queue silently ?
        async with self._status_lock:
            if self._status == NodeActiveStatus.DISCONNECTED:
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Node is not active. Last communication {self._last_request}")

        try:
            await self._queue.put(message)
        except Exception as exp:
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: Can't send message to the client. Exception: {exp}")

    def set_context(self, context):
        """Sets context for the current RPC call

        Args:
            context: RPC call context
        """
        self.context = context
        self.context.add_done_callback(self._on_get_task_request_done)


    def get(self) -> asyncio.coroutine:
        """Get tasks assigned by the main thread

        !!! note "Returns coroutine"
            This function return an asyncio coroutine. Please use `await` while calling.

        """
        return self._queue.get()

    def _on_get_task_request_done(self, context: grpc.aio.ServicerContext) -> None:
        """Callback to execute each time RPC call is completed

        The callback is executed when the RPC call is canceled, done or aborted, including
        if the process on the node side stops.
        """
        self._status = NodeActiveStatus.WAITING

        # Imply DISCONNECT after 10seconds rule asynchronously
        if self._status_task is None:
            self._status_task = asyncio.create_task(self._change_node_status_disconnected())


    async def _change_node_status_disconnected(self) -> None:
        """Updates node status as `DISCONNECTED`

        Node becomes DISCONNECTED if it doesn't become ACTIVE in 10 seconds
        """

        # Sleep at least 10 seconds in WAITING
        await asyncio.sleep(10)

        # If the status still WAITING set status to DISCONNECTED
        async with self._status_lock:
            if self._status == NodeActiveStatus.WAITING:
                self._status = NodeActiveStatus.DISCONNECTED
                logger.warning(
                    f"Node {self._id} is disconnected. Request/task that are created "
                    "for this node will be flushed" )
            # TODO: clean the queue


class AgentStore:
    """Stores node agents"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Constructor of the agent store

        Args:
            loop: asyncio event loop that research server runs. Agent store should use
                same event loop for async operations
        """
        self._node_agents: NodeAgent = {}

        self._loop = loop
        self._store_lock = asyncio.Lock()

    async def retrieve(
            self,
            node_id: str
    ) -> NodeAgent:
        """Retrieves a node agent for a given node ID.

        Depending if this node is already known to the store this method gets existing agent or.
        registers a new agent.

        Args:
            node_id: ID of receiving node

        Return:
            The node agent to manage tasks that are assigned to it.
        """
        # Lock during all sequence to ensure atomicity
        async with self._store_lock:
            node = self._node_agents.get(node_id)
            if not node:
                node = NodeAgent(id=node_id, loop=self._loop)
                self._node_agents.update({node_id: node})

        return node

    async def get_all(self) -> Dict[str, NodeAgent]:
        """Returns all node agents regardless of their status (ACTIVE, DISCONNECTED, ...).

        Returns:
            Dictionary of node agent objects, by node ID
        """

        async with self._store_lock:
            # a shallow copy is wanted so that
            # - we have a distinct (stable) list of NodeAgents that can be processed in calling func
            # - we use same NodeAgents objects (not a copy)
            return copy.copy(self._node_agents)

    async def get(
            self,
            node_id: str
    ) -> Optional[NodeAgent]:
        """Gets node agent by given node id

        Args:
            node_id: Id of the node, or None if no agent exists for this node ID
        """
        async with self._store_lock:
            return self._node_agents.get(node_id)
