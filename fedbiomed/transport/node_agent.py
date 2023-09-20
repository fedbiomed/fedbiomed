import asyncio
import threading
import concurrent
import contextlib
import grpc 

from enum import Enum
from typing import Callable, Any, List
from datetime import datetime

from fedbiomed.common.message import Message
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_method_spec



_node_store_lock = threading.Lock()



_pool = concurrent.futures.ThreadPoolExecutor()


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



# Combines threading.Lock with asyncio
# See source: https://stackoverflow.com/a/63425191/6111150 
@contextlib.asynccontextmanager
async def async_thread_lock(lock):
    loop = asyncio.get_event_loop()

    await loop.run_in_executor(_pool, lock.acquire)

    try:
        yield
    finally:
        lock.release()


class NodeAgent:

    def __init__(
            self, 
            id: str, 
            loop,
        ):
        """Represent the client that connects to gRPC server"""
        self.id: str = id 
        self.last_request: datetime = None 

        # Node should be active when it is first instantiated
        self.status: NodeActiveStatus = NodeActiveStatus.ACTIVE

        self._queue = asyncio.Queue()
        self._loop = loop 
        self._status_task = None
        self.status_lock = asyncio.Lock()

    async def active(self) -> None:
        """Updates node status as active"""

        async with self.status_lock:

            # Inform user that node is online again
            if self.status == NodeActiveStatus.DISCONNECTED:
                logger.info(f"Node {self.id} is back online!")

            self.status = NodeActiveStatus.ACTIVE
                    
            # Cancel status task if there is any running
            if self._status_task:
                self._status_task.cancel()

    async def send_(self, message: Message):
        """Async function send message to researcher"""

        if not isinstance(message, Message):
            raise Exception("Message is not an instance of fedbiomed.common.message.TaskMessage")


        async with self.status_lock:
            if self.status == NodeActiveStatus.DISCONNECTED:
                raise Exception(f"Node is not active. Last communication {self.last_request}")

        try:
            await self._queue.put(message)
        except Exception as exp:
            raise Exception(f"Can't send message to the client. Exception: {exp}")


    def get_status_threadsafe(self):
        """Gets node status as threadsafe 
        
        Node status should be accessed using this method since it can be 
        modified by asyncio thread. 
        """
        async def get():
            async with self.status_lock:
                return self.status

        future = asyncio.run_coroutine_threadsafe(get(), self._loop)
        return future.result()

    

    def set_context(self, context):
        """Sets context for the current RPC call
        
        Args:
            context: RPC call context
        """
        self.context = context
        self.context.add_done_callback(self._on_get_task_request_done)


    def send(self, message: Message) -> None:
        """Send task to the client
        
        !!! warning "Important"
            You can send only the message that defined within the scope of `TaskResponse`
            please see Message class
        
        Args: 
            message: The task that is going to be sent to the node
            callback: Callback to execute once the task reply is arrived
        """

        try:
            return asyncio.run_coroutine_threadsafe(self.send_(message), self._loop)
        except Exception as exp:
            raise Exception(f"Can't send message to the client. Exception: {exp}")
        

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
        self.status = NodeActiveStatus.WAITING

        # Imply DISCONNECT after 10seconds rule asynchronously 
        self._status_task = asyncio.create_task(self._change_node_status_disconnected())



    async def _change_node_status_disconnected(self) -> None:
        """Updates node status as `DISCONNECTED`
        
        Node becomes DISCONNECTED if it doesn't become ACTIVE in 10 seconds
        """

        # Sleep at least 10 seconds in WAITING
        await asyncio.sleep(10)

        # If the status still WAITING set status to DISCONNECTED
        async with self.status_lock:
            if self.status == NodeActiveStatus.WAITING:
                self.status = NodeActiveStatus.DISCONNECTED
                logger.warning(
                    f"Node {self.id} is disconnected. Request/task that are created "
                    "for this node will be flushed" )
            # TODO: clean the queue


class AgentStore:
    """Stores node agents"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Constructs agent store
        
        Args: 
            loop: asyncio event loop that research server runs. Agent store should use
                same event loop for async operations
        """
        self.node_agents: NodeAgent = {}

        self._loop = loop
        self.store_lock = asyncio.Lock()

    async def get_or_register(
            self, 
            node_id:str
        ) -> NodeAgent:
        """Registers or gets node agent. 

        Depending of the state this method registers or gets new NodeAgent. 
        
        Args: 
            node_id: ID of receiving node 

        Return:
            The node agent to manage tasks that are assigned to it.
        """

        node = await self.get(node_id=node_id)

        if not node:
            node = await self.register(node_id)

        await node.active()

        return node


    async def register(
            self, 
            node_id:str
        ) -> NodeAgent:
        """Register new node agent. 
        
        Args: 
            node_id: ID to register
        """
        # Lock the thread for register operation
        node = NodeAgent(id=node_id, loop=self._loop)
        async with self.store_lock:
            self.node_agents.update({node_id: node})

        return node
        
    async def get_all(self) -> List[NodeAgent]:
        """Returns all node agents regardless ACTIVE or DISCONNECTED"""

        async with self.store_lock:
            return self.node_agents
        
    async def get(
            self,
            node_id: str
        ) -> NodeAgent:
        """Gets node agent by given node id
        
        Args: 
            node_id: Id of the node
        """
        async with self.store_lock:
            return self.node_agents.get(node_id)