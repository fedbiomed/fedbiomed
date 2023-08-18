 

import asyncio
import grpc
import queue
import threading
import sys 
import signal 

from typing import Callable, Dict

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.proto.researcher_pb2 import TaskRequest, FeedBackMessage
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer

import uuid
import time

import statistics

class GRPCStop(Exception):
    """Stop exception for gRPC client"""
    pass 

class GRPCTimeout(Exception):
    """gRPC timeout error"""
    pass 


class GRPCStreamingKeepAliveExceed(Exception):
    """"Task reader keep alive error"""
    pass

logger.setLevel("DEBUG")
NODE_ID = str(uuid.uuid4())




DEFAULT_ADDRESS = "localhost:50051"
STREAMING_MAX_KEEP_ALIVE_SECONDS = 60 


def create_channel(
    address: str = DEFAULT_ADDRESS ,
    certificate: str = None
) -> grpc.Channel :
    """ Create gRPC channel 
    
    Args: 
        ip: 
        port:
        certificate:
    
    Returns: 
        gRPC connection channel
    """
    channel_options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]

    if certificate is None: 
        channel = grpc.aio.insecure_channel(address, options=channel_options)
    else:
        # TODO: Create secure channel
        pass
    
    # TODO: add callback fro connection state

    return channel



async def task_reader_unary(
        stub, 
        node: str, 
        event,
        callback: Callable = lambda x: x,
         
):
    """Task reader as unary RPC
    
    This methods send unary RPC to gRPC server (researcher) and get tasks 
    in a stream. Stream is used in order to receive larger messages (more than 4MB)
    After a task received it sends another task request immediately. 

    Args: 
        stub: gRPC stub to execute RPCs. 
        node: Node id 
        callback: Callback function to execute each time a task received
    """
    #queue = asyncio.Queue()
    task_queue = queue.Queue()

    def cancel_request(unused_signum, unused_frame):
        task.cancel()
        sys.exit(0)

    
    while True:
        
        if event.is_set():
            raise GRPCStop("gRPC has been stooped")


        task = stub.GetTaskUnary(
            TaskRequest(node=f"{node}")
        )

        # Cancelation
        signal.signal(signal.SIGINT, cancel_request)

        # Prepare reply
        reply = bytes()
        async for answer in task:
            
            reply += answer.bytes_

            if answer.size != answer.iteration:
                continue
            else:
                # Execute callback
                callback(
                    Serializer.loads(reply)
                    )
                # Reset reply
                reply = bytes()


async def task_reader(
        stub, 
        node: str, 
        callback: Callable = lambda x: x
        ) -> None:
        """Low-level task reader implementation 

        This methods launches two coroutine asynchronously:
            - Task 1 : Send request stream to researcher to get tasks 
            - Task 2 : Iterates through response to retrieve tasks  

        Task request iterator stops until a task received from the researcher server. This is 
        managed using asyncio.Condition. Where the condition is "do not ask for new task until receive one"

        Once a task is received reader fires the callback function

        Args: 
            node: The ID of the node that requests for tasks
            callback: Callback function that takes a single task as an arguments

        Raises:
            GRPCTimeout: Once the timeout is exceeded reader raises time out exception and 
                closes the streaming.
        """  
        event = asyncio.Event()

        async def request_tasks(event):
            """Send request stream to researcher server"""
            async def request_():
                # Send starting request without relying on a condition
                count  = 1
                logger.info("Sending first request after creating a new stream connection")
                state.n = time.time()
                yield TaskRequest(node=f"{count}---------{node}")
                                
                while True:
                    # Wait before getting answer from previous request
                    await event.wait()
                    event.clear()

                    #logger.info(f"Sending another request ---- within the stream")   
                    state.n = time.time()  
                    yield TaskRequest(node=f"{count}---------{node}")
                    


            # Call request iterator withing GetTask stub
            state.task_iterator = stub.GetTask(
                        request_()
                    )

        async def receive_tasks(event):
            """Receives tasks form researcher server """
            async for answer in state.task_iterator:
                
                reply += answer.bytes_
                # print(f"{answer.size}, {answer.iteration}")
                if answer.size != answer.iteration:
                    continue
                else:
                    event.set()
                    callback(Serializer.loads(reply))
                    reply = bytes()
                    
                
        # Shared state between two coroutines
        state = type('', (), {})()

        await asyncio.gather(
            request_tasks(event), 
            receive_tasks(event)
            )





class ResearcherClient:
    """gRPC researcher component client 
    
    Attributes: 


    """
    def __init__(
            self,
            handler = None,
            certificate: str = None,
        ):


        # TODO: implement TLS 
        if certificate is not None:
            # TODO: create channel as secure channel 
            pass 
        
        self._client_registered = False
        self._stop_event = threading.Event()
        self._client_thread = threading.Thread()

    async def connection(self, event):
        """Create long-lived connection with researcher server"""
        
        self._feedback_channel = create_channel(certificate=None)
        self._log_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)

        self._task_channel = create_channel(certificate=None)
        self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)

        logger.info("Waiting for researcher server...")

        # Starts loop to ask for
        await self.get_tasks(event)   
            

    async def get_tasks(self, event):

        while True:
            logger.info("Sending new task request")
            try:
                # await task_reader(stub= self._stub, node=NODE_ID, callback=self.on_task)
                await task_reader_unary(stub= self._stub, node=NODE_ID, event=event, callback= lambda x: x)

            except GRPCStop:
                # Break gRPC while loop
                logger.info("Shutting down node gRPC client... ")
                break

            except grpc.aio.AioRpcError as exp:
                print(exp.code())
                if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("Stream TIMEOUT Error")
                    await asyncio.sleep(2)
            
                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                    await asyncio.sleep(2)
                else:
                    raise Exception("Request streaming stopped ") from exp
            finally:
                pass
            
    
    async def send_log(self, log):

        _ = await self._log_tub.Feedback(Log(log=log))


    def start(self):
        """Starts researcher gRPC client"""

        self._stop_event = threading.Event()

        # Runs gRPC async client 
        def run(event):
            try: 
                asyncio.run(
                        self.connection(event), debug=False
                    )
            except KeyboardInterrupt:
                asyncio.get_running_loop().close()

        
        t = threading.Thread(target=run, args=(self._stop_event,))
        t.start()

    def stop(self):
        """Stop gently running asyncio loop and its thread"""

        self._stop_event.set()


if __name__ == '__main__':
    
    ResearcherClient().start()