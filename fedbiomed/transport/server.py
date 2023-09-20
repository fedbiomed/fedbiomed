
import asyncio
import grpc
import threading
import ctypes
import signal


from abc import abstractmethod
from typing import Callable, Iterable
from google.protobuf.message import Message as ProtoBufMessage


from fedbiomed.transport.protocols.researcher_pb2 import Empty
import fedbiomed.transport.protocols.researcher_pb2_grpc as researcher_pb2_grpc


from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, TaskResponse, TaskRequest, FeedbackMessage
from fedbiomed.common.constants import MessageType
from fedbiomed.transport.node_agent import AgentStore

from concurrent import futures
import time
import random 
import json 
import sys


DEFAULT_PORT = 50051
DEFAULT_HOST = 'localhost'

logger.setLevel("INFO")

rnd = random.sample(range(0, 1000000), 100000)

training_task = {
    "researcher_id" : "test-researcher",
    "job_id" : "test-job",
     "training_args" : {"args_1": 12, "args_2": True, "args_3": "string"},
     "model_args": {"args_1": 12, "args_2": True, "args_3": "string"},
    "dataset_id": "ddd-id",
    "training": True, 
    "training_plan": "Basic training plan",
    "secagg_servkey_id": None,
    "secagg_biprime_id ": None,
    "secagg_random": None, 
    "secagg_clipping_range":  None,
    "round": 1,
    "aggregator_args": {"args_1": 12, "args_2": True, "args_3": "string"},
    "aux_vars": {"arg_1": rnd, "args_2": rnd, "args_3": rnd, "args_4": rnd, "args_5": rnd, "args_6": rnd, "args_7": rnd,
                 "arg_1x": rnd, "args_2x": rnd, "args_3x": rnd, "args_4x": rnd, "args_5x": rnd, "args_6x": rnd, "args_7x": rnd,
                 "arg_1y": rnd, "args_2y": rnd, "args_3y": rnd, "args_4y": rnd, "args_5y": rnd, "args_6y": rnd, "args_7y": rnd,
                 "arg_1z": rnd, "args_2z": rnd, "args_3z": rnd, "args_4z": rnd, "args_5z": rnd, "args_6z": rnd, "args_7z": rnd,
                 "arg_1d": rnd, "args_2d": rnd, "args_3d": rnd, "args_4d": rnd, "args_5d": rnd, "args_6d": rnd, "args_7d": rnd
                  },
    "model_params": {"arg_1": rnd, "args_2": rnd, "args_3": rnd, "args_4": rnd, "args_5": rnd, "args_6": rnd, "args_7": rnd },

}


small_task = {
        "researcher_id" : "test-researcher",
    "job_id" : "test-job",
     "training_args" : {"args_1": 12, "args_2": True, "args_3": "string"},
     "model_args": {"args_1": 12, "args_2": True, "args_3": "string"},
    "dataset_id": "ddd-id",
    "training": True, 
    "training_plan": "Basic training plan",
    "secagg_servkey_id": None,
    "secagg_biprime_id ": None,
    "secagg_random": None, 
    "secagg_clipping_range":  None,
    "round": 1,
}


# Max message length as bytes
MAX_MESSAGE_BYTES_LENGTH = 4000000 - sys.getsizeof(bytes("", encoding="UTF-8")) # 4MB 


class ResearcherServicer(researcher_pb2_grpc.ResearcherServiceServicer):
    """RPC Servicer """

    def __init__(
            self, 
            agent_store: AgentStore, 
            on_message: Callable
        ) -> None:
        """Constructor of gRPC researcher servicer
        
        Args: 
            agent_store: The class that stores node agents 
            on_message: Callback function to execute once a message received from the nodes
        """
        super().__init__()
        self.agent_store = agent_store
        self._on_message = on_message


    async def GetTaskUnary(
            self, 
            request: ProtoBufMessage, 
            context: grpc.aio.ServicerContext
        ) -> None:
        """Gets unary RPC request and return stream of response 
        
        Args: 
            request: RPC request
            context: RPC peer context
        """
        
        task_request = TaskRequest.from_proto(request).get_dict()
        logger.debug(f"Node: {task_request.get('node')} polling for the tasks")
         
        node_agent = await self.agent_store.get_or_register(node_id=task_request["node"])
        # Update node active status as active
        node_agent.set_context(context)
            
        task = await node_agent.get()
        task = Serializer.dumps(task.get_dict())

        chunk_range = range(0, len(task), MAX_MESSAGE_BYTES_LENGTH)    
        for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
            stop = start + MAX_MESSAGE_BYTES_LENGTH 
            
            yield TaskResponse(
                size=len(chunk_range),
                iteration=iter_,
                bytes_=task[start:stop]
            ).to_proto()


    async def ReplyTask(
            self, 
            request_iterator: Iterable[ProtoBufMessage], 
            unused_context: grpc.aio.ServicerContext
        ) -> None:
        """Gets stream replies from the nodes
        
        Args: 
            request_iterator: Iterator for streaming 
            unused_context: Request service context
        """
        
        reply = bytes()
        async for answer in request_iterator:
            reply += answer.bytes_
            if answer.size != answer.iteration:
                continue
            else:
                # Deserialize message
                message = Serializer.loads(reply)
                self._on_message(message, MessageType.REPLY)
                reply = bytes()

        return Empty()


    async def Feedback(
            self, 
            request: ProtoBufMessage, 
            unused_context: grpc.aio.ServicerContext
        ) -> None:
        """Executed for Feedback request received from the nodes
        
        Args:
            request: Feedback message  
            unused_context: Request service context
        """

        # Get the type of Feedback | log or scalar
        one_of = request.WhichOneof("feedback_type")
        feedback = FeedbackMessage.from_proto(request)

        # Execute on message assigned by the researcher.requests modules
        self._on_message(feedback.get_param(one_of), MessageType.convert(one_of))
        
        return Empty()



def _default_callback(self, x):
    print(f'Default callback: {type(x)} {x}')


class _GrpcAsyncServer:
    """GRPC Server class"""

    def __init__(
            self, 
            debug: bool = False, 
            on_message: Callable = _default_callback
        ) -> None:
        """Constructs GrpcServer 
        
        Args: 
            debug: Activate debug mode for gRPC asyncio
            on_message: Callback function to execute once a message received from the nodes
        """
        
        self._server = None 
        self._thread = None 
        self._debug = debug
        self._on_message = on_message
        self._loop = None


    async def start(self):

        self._loop = asyncio.get_event_loop()

        self._server = grpc.aio.server( 
           # futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ])
    
        self.agent_store = AgentStore(loop=asyncio.get_event_loop())

        researcher_pb2_grpc.add_ResearcherServiceServicer_to_server(
            ResearcherServicer(
                agent_store=self.agent_store,
                on_message=self._on_message), 
                server=self._server
            )
        
        self._server.add_insecure_port(DEFAULT_HOST + ':' + str(DEFAULT_PORT))
        
        

        await self._server.start()

        try:
            if self._debug: 
                logger.debug("Waiting for termination")
            await self._server.wait_for_termination()
        finally:
            if self._debug: 
                logger.debug("Done starting the server")


    async def broadcast(self, message: Message):
        """Broadcasts given message to all active clients"""

        agents = await self.agent_store.get_all()
        for _, agent in agents:
            if agent.active == False:
                logger.info(f"Node {agent.id} is not active")
            agent.send(message)
            agents.append(agent.id)      

        return agents

    async def get_agent(self, node_id: str):

        async with self._agent_store_lock:
            self.agent_store.get(node_id)

    

class GrpcServer(_GrpcAsyncServer):
    
    def start(self):
        """Stats async GrpcServer """

        def run():
            try:
                asyncio.run(super().start())
            except Exception as e:
                logger.error("Researcher gRPC server has stopped. Please try to restart your kernel")
            

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

        # FIXME: This implementation assumes that nodes will be able connect in 3 seconds
        logger.info("Starting researcher service...")   
        time.sleep(3)

    def broadcast(self, message: Message):
        """Broadcast message
        
        !!! warning "Important"
            This method should be called only from main thread.

        Args:
            message: Message to broadcast
        """

        return self._run_threadsafe(super().broadcast(message))

    def is_alive(self) -> bool:
        """Checks if the thread running gRPC server still alive
        
        Return:
            gRPC server running status
        """
        return False if not isinstance(self._thread, threading.Thread) else self._thread.is_alive()

    def get_agent(self, node_id: str):
        """Gets node agent by node id
        
        Args:
            node_id: Id of the node
        """
        return self._run_threadsafe(super().agent_store.get(node_id))
    
    def get_all_agents(self):
        """Gets all agents from agent store"""

        return self._run_threadsafe(super().agent_store.get_all())


    def _run_threadsafe(self, coroutine):
        """Runs given coroutine threadsafe
        
        Args:
            coroutine: Awaitable function to be executed as threadsafe
        """

        future = asyncio.run_coroutine_threadsafe(
            coroutine, self._loop
        )

        return future.result()

if __name__ == "__main__":

    def handler(signum, frame):
        print(f"Node cancel by signal {signal.Signals(signum).name}")
        sys.exit(1)
        
    rs = GrpcServer(debug=True)
    signal.signal(signal.SIGHUP, handler)

    try:
        rs.start()
        while rs.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Researcher cancel by keyboard interrupt")

