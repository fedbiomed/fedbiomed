
import asyncio
import grpc
import threading
import ctypes
import signal
import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc

from typing import Callable

from fedbiomed.proto.researcher_pb2 import Empty

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


class ServerStop(Exception):
    pass 

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

    def __init__(
            self, 
            agent_store: AgentStore, 
            on_message: Callable
        ) -> None:
        """Constructor of gRPC researcher servicer"""
        super().__init__()
        self.agent_store = agent_store
        self._on_message = on_message


    async def GetTask(self, request_iterator, context):
        """Stream-Stream get task service 
        
        !!! "note"
            It is not used currently please see GetTaskUnary
        """
            

        async for req in request_iterator:
            
            # print("Request has arrived")
            # print(req)
            node = req.node

            task_request = TaskRequest.from_proto(req)
            logger.info(f"Received request form {task_request.get('node')}")
        
            node_agent = await self.agent_store.get_or_register(node_id=task_request["node"])
            node_agent.set_context(context)
            # Here call task and wait until there is no

            logger.info(f"Received request form {node}")
            # Dumps the message
            await asyncio.sleep(5)
            task = Serializer.dumps(small_task)
            chunk_range = range(0, len(task), MAX_MESSAGE_BYTES_LENGTH)
            for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
                stop = start + MAX_MESSAGE_BYTES_LENGTH 
                yield TaskResponse(
                    size=len(chunk_range),
                    iteration=iter_,
                    bytes_=task[start:stop]
                ).to_proto()

            logger.info(f"Request finished for {node}")


    async def GetTaskUnary(self, request, context):
        """Gets unary RPC request and return stream of response 
        
        Args: 
            request: RPC request
            context: RPC peer context
        """

        task_request = TaskRequest.from_proto(request).get_dict()
        logger.info(f"Received request form {task_request.get('node')}")
         
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

    async def ReplyTask(self, request_iterator, context):
        """Gets stream replies from the nodes"""
        
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


    async def Feedback(self, request, unused_context):
        
        one_of = request.WhichOneof("feedback_type")
        feedback = FeedbackMessage.from_proto(request)
        self._on_message(feedback.get_param(one_of), MessageType.convert(one_of))
        
        return Empty()



def _default_callback(self, x):
    print(f'Default callback: {type(x)} {x}')



class GrpcServer:

    def __init__(self, debug: bool = False, on_message: Callable = _default_callback) -> None:
        self._server = None 
        self._thread = None 
        self._debug = debug
        self._on_message = on_message 


    async def _start(self):

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
            if self._debug: print("_start: done starting server")
            while await self._server.wait_for_termination(timeout=1):
                if self._debug: print("_start: loop wait_for_termination")
        finally:
            if self._debug: print("_start: finally")

    def broadcast(self, message: Message):
        """Broadcasts given message to all active clients"""

        agents = []
        for _, agent in self.agent_store.node_agents.items():
            if agent.active == False:
                logger.info(f"Node {agent.id} is not active")
            agent.send(message)
            agents.append(agent.id)       
        return agents

    def start(self):

        def run():
            try:
                asyncio.run(
                    self._start()
                )
            except ServerStop:
                if self._debug:
                    print("Run: caught user stop exception")
            finally:
                if self._debug:
                    print("Run: finally")

        self._thread = threading.Thread(target=run)
        self._thread.start()

        # Sleep 2 seconds before releasing READY event
        # FIXME: This implementation assumes that nodes will be able connect in 3 seconds
        logger.info("Starting researcher service...")   
        time.sleep(3)

    def stop(self):
        """Stops researcher server"""
        
        #future = asyncio.run_coroutine_threadsafe(self._stop(), self._loop)
        #future.result(timeout=5)
        print("stop: after future")

        if not isinstance(self._thread, threading.Thread):
            stopped_count = 0
        else:
            stopped_count = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._thread.ident),
                                                                       ctypes.py_object(ServerStop))
        if stopped_count != 1:
            logger.error("stop: could not deliver exception to thread")
        else:
            self._thread.join()
        if self._debug: print("stop: finishing")

    def is_alive(self) -> bool:
        return False if not isinstance(self._thread, threading.Thread) else self._thread.is_alive()


if __name__ == "__main__":

    def handler(signum, frame):
        print(f"Node cancel by signal {signal.Signals(signum).name}")
        rs.stop()
        sys.exit(1)
        
    rs = GrpcServer(debug=True)
    signal.signal(signal.SIGHUP, handler)

    try:
        rs.start()
        while rs.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Researcher cancel by keyboard interrupt")
        try:
            rs.stop()
        except KeyboardInterrupt:
            print("Immediate keyboard interrupt, dont wait to clean")

