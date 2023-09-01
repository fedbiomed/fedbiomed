import asyncio
import threading
import concurrent
import contextlib

from typing import Callable, Any, List
from datetime import datetime

from fedbiomed.common.message import Message
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_method_spec

_global_lock = threading.RLock()


_pool = concurrent.futures.ThreadPoolExecutor()


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
            ip: str,
            loop,
        ):
        """Represent the client that connects to gRPC server"""
        self.id: str = id 
        self.ip: str = ip
        self.last_request: datetime = None 
        self.active: bool = None
        self._queue = asyncio.Queue()
        self._loop = loop 

    def send(self, message: Message) -> None:
        """Send task to the client
        
        !!! warning "Important"
            You can send only the message that defined within the scope of `TaskResponse`
            please see Message class
        
        Args: 
            message: The task that is going to be sent to the node
            callback: Callback to execute once the task reply is arrived
        """


        print("Sending request!")
        # TODO: The messaged that are going to send to node 
        # should be declared as Task to have task id automatically 
        # created. However, this implementation should change once
        # the messages are typed on gRPC layer instead serializing 
        # the message string and send as researcher_pb2.TaskResponse
        # please see researcher.proto file
        if not isinstance(message, Message):
            raise Exception("Message is not an instance of fedbiomed.common.message.TaskMessage")


        if not self.active:
            raise Exception(f"Node is not active. Last communication {self.last_request}")
         
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, message
            )
        except Exception as exp:
            raise Exception(f"Can't send message to the client. Exception: {exp}")
        

    def register_callback(self, task_id: str, callback: Callable):
        """Registers callback for the task 
        
        Callback function is executed once the task reply has arrived.
        """

        spec = get_method_spec(callback)

        if len(spec) != 2: 
            raise Exception("Callback for reply should have at 2 arguments. First argument " 
                            "for reply second argument for node id")
        
        self._task_callbacks.update({
            task_id: callback
        })

    def get(self) -> asyncio.coroutine:
        """Get tasks assigned by the main thread
        
        !!! note "Returns coroutine"
            This function return an asyncio coroutine. Please use `await` while calling.

        """
        return self._queue.get()
    

class AgentStore:
    """Stores node agents"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Constructs agent store
        
        Args: 
            loop: asyncio event loop that research server runs. Agent store should use
                same event loop for async operations
        """
        self._loop = loop
        self.node_agents = {}

    async def get_or_register(
            self, 
            node_id:str, 
            node_ip: str
        ) -> NodeAgent:
        """Registers or gets node agent. 

        Depending of the state this method registers or gets new NodeAgent. 
        
        Args: 
            node_id: ID of receiving node 
            node_ip: IP of receiving node

        Return:
            The node agent to manage tasks that are assigned to it.
        """

        node = self.get(node_id=node_id)
        
        # If node is existing return immediately by updating active status
        if node:
            node.active = True
            return node 
        
        # Register new NodeAgent
        result = await self._loop.run_in_executor(
            None, self.register,  node_id, node_ip)
        
        return result


    def register(
            self, 
            node_id:str, 
            node_ip: str
        ) -> NodeAgent:
        """Register new node agent. 
         
        This method designed to be coroutine and thread-safe
        
        Args: 
            node_id: ID to register
            node_ip: IP to register
        """
        # Lock the thread for register operation
        try:
            _global_lock.acquire()
        
        finally:
            node = NodeAgent(id=node_id, ip=node_ip, loop=self._loop)
            self.node_agents.update({
                node_id: node
            })
            node.active = True
            _global_lock.release()

        return node
        

    def get(
            self,
            node_id: str
        ) -> NodeAgent:
        """Gets node agent by given node id
        
        Args: 
            node_id: Id of the node
        """
        return self.node_agents.get(node_id)