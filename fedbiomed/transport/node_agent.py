import asyncio
import threading
import concurrent
import contextlib

from typing import Callable, Any, List
from datetime import datetime

from fedbiomed.common.message import TaskMessage
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
        self._task_callbacks = {}
        self._loop = loop 

    def send(self, message: TaskMessage, callback: Callable = None):
        """Send task to the client
        
        !!! warning "Important"
            You can send only the message that defined within the scope of `TaskResponse`
            please see Message class
        
        Args: 
            message: The task that is going to be sent to the node
            callback: Callback to execute once the task reply is arrived
        """

        # TODO: The messaged that are going to send to node 
        # should be declared as Task to have task id automatically 
        # created. However, this implementation should change once
        # the messages are typed on gRPC layer instead serializing 
        # the message string and send as researcher_pb2.TaskResponse
        # please see researcher.proto file
        if not isinstance(message, TaskMessage):
            raise Exception("Message is not an instance of fedbiomed.common.message.TaskMessage")


        if not self.active:
            raise Exception(f"Node is not active. Last communication {self.last_request}")
         
        if callback is not None:
            self.register_callback(message.param("task_id"), callback)
        
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, message
            )
            
            print(self._queue)
        except Exception as exp:
            raise Exception("Can't send message to the client")
        
        return True

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


    def reply(self, reply: TaskMessage):

        # Execute callback 
        task_id = reply.param('task_id')

        if task_id not in self._task_callbacks:
            logger.debug(f"No callback registered for the task {task_id}")
            return 
        
        self._task_callbacks[task_id](reply, self.id)


    def get(self):
        """Get tasks assigned by the main thread"""
        return self._queue.get()
    

class AgentStore:

    def __init__(self, loop):
        self._loop = loop
        self.node_agents = {}

    async def get_or_register(self, node_id:str, node_ip: str) -> NodeAgent:
        """Registers or gets node agent 
        
        Args: 

        """

        node = self.get(node_id=node_id)
        if node:
            return node 
        
        print("Here")

        result = await self._loop.run_in_executor(
            None, self.register,  node_id, node_ip)
        
        print("And here")
        return result


    def register(self, node_id:str, node_ip: str) -> NodeAgent:
        
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
        

    def get(self, node_id: str) -> NodeAgent:
        """Gets node agent by given node id
        
        Args: 
            node_id: Id of the node
        """
        return self.node_agents.get(node_id)