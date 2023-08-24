 

import asyncio
import concurrent.futures
import grpc
import queue
import threading
import sys 
import signal 
import ctypes

from typing import Callable, Dict, Union
from google.protobuf.message import Message as ProtobufMessage

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.proto.researcher_pb2 import TaskRequest, FeedbackMessage
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH
from fedbiomed.common.message import Message, TaskRequest, FeedbackMessage, TaskResult

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


class ClientStop(Exception):
    pass 

logger.setLevel("DEBUG")
NODE_ID = str(uuid.uuid4())






DEFAULT_ADDRESS = "localhost:50051"
#STREAMING_MAX_KEEP_ALIVE_SECONDS = 60 



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
        ("grpc.keepalive_time_ms", 1000 * 2)
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
    pass 


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
                yield TaskRequest(node=node).to_proto()
                                
                while True:
                    # Wait before getting answer from previous request
                    await event.wait()
                    event.clear()

                    #logger.info(f"Sending another request ---- within the stream")   
                    state.n = time.time()  
                    yield TaskRequest(node=node).to_proto()
                    


            # Call request iterator withing GetTask stub
            state.task_iterator = stub.GetTask(
                        request_()
                    )

        async def receive_tasks(event):
            """Receives tasks form researcher server """
            reply = bytes()
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
            on_message: Callable = None,
            certificate: str = None,
            debug: bool = False
        ):


        # TODO: implement TLS 
        if certificate is not None:
            # TODO: create channel as secure channel 
            pass 
        
        self._send_queue = asyncio.Queue()
        self._client_registered = False
        self.on_message = on_message or self._default_callback
        self._thread = None
        self._thread_loop = None

        self._debug = debug

    def _default_callback(self, x):
        print(f'Default callback: {type(x)} {x}')


    async def connection(self, debug: bool = False):
        """Create long-lived connection with researcher server"""
        
        self._feedback_channel = create_channel(certificate=None)
        self._feedback_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)
        
        self._task_channel = create_channel(certificate=None)
        self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)

        self._thread_loop = asyncio.get_running_loop()

        logger.info("Waiting for researcher server...")

        await asyncio.gather(
            self.get_tasks(debug),
            self.send_queue_listener(debug),
        )
            
    async def send_queue_listener(self, debug: bool = False):
        """Listens queue that contains message to send to researcher """

        #while not SHUTDOWN_EVENT.is_set():
        while True: 
            #try:
            #    msg = self._send_queue.get_nowait()
            #except asyncio.queues.QueueEmpty:
            #    await asyncio.sleep(0.5)
            #    continue
            msg = await self._send_queue.get()

            # If it is aUnary-Unary RPC call
            if isinstance(msg["stub"], grpc.aio.UnaryUnaryMultiCallable):
                await msg["stub"](msg["message"])
            elif isinstance(msg["stub"], grpc.aio.grpc.aio.StreamUnaryMultiCallable): 
                reply = Serializer.dumps(msg["message"].get_dict())
                chunk_range = range(0, len(reply), MAX_MESSAGE_BYTES_LENGTH)
                for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
                    stop = start + MAX_MESSAGE_BYTES_LENGTH 
                    yield await TaskResult(
                        size=len(chunk_range),
                        iteration=iter_,
                        bytes_=reply[start:stop]
                    ).to_proto()

    async def get_tasks(self, debug: bool = False):
        """Long-lived polling to request tasks from researcher."""

        #while not SHUTDOWN_EVENT.is_set():
        while True:
            logger.info("Sending new task request")
            try:
                # await task_reader(stub= self._stub, node=NODE_ID, callback=self.on_message)
                # await task_reader_unary(stub= self._stub, node=NODE_ID, callback= lambda x: x)
                while True:
                    logger.info("Sending new task request")
                    print(self._task_channel.get_state())
                    self.__request_task_iterator = self._stub.GetTaskUnary(
                        TaskRequest(node=f"{NODE_ID}").to_proto(), timeout=60,
                    )

                    # Prepare reply
                    reply = bytes()
                    async for answer in self.__request_task_iterator:
                        reply += answer.bytes_
                        if answer.size != answer.iteration:
                            continue
                        else:
                            # Execute callback
                            self.on_message(Serializer.loads(reply))
                            # Reset reply
                            reply = bytes()
            
            except grpc.aio.AioRpcError as exp:
                if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("Stream TIMEOUT Error")
                    print(exp)
                    await asyncio.sleep(2)
            
                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    print(self._task_channel.get_state())
                    logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                    await asyncio.sleep(2)
                else:
                    if debug: print(f'get_tasks: unknown exception {exp.__class__.__name__} {exp}')
                    raise Exception("Request streaming stopped ") from exp
            finally:
                pass
            

    def _create_send_task(
            self,
            stub: Callable, 
            message: ProtobufMessage
        ) -> Dict[str, Union[Callable, ProtobufMessage]]:
        """Validates and create object tpo add queue of sending messages 
        
        Args: 
            stub: RPC stub to execute 
            message: Message to send researcher
        
        Returns:
            Contains stub and message to add into the queue
        """

        if not isinstance(stub, Callable):
            raise Exception("'stub' must an instance of ResearcherServiceStub")
        
        if not isinstance(message, ProtobufMessage):
            raise Exception("'message' should be be an instance of Protobuf")

        return {"stub": stub, "message": message}
    
    async def _send_from_thread(self, rpc, message: ProtobufMessage):
        try:
            #await asyncio.sleep(5)
            await self._send_queue.put(
                self._create_send_task(rpc,message)
            )
            return True
        except asyncio.CancelledError:
            if self._debug: print("_send_from_thread: cancelled send task")
            return False


    def _run_thread_safe(self, coroutine: asyncio.coroutine):
        """Runs given coroutine as thread-safe"""

        if not isinstance(self._thread_loop, asyncio.AbstractEventLoop):
            raise Exception("send: the thread loop is not ready")

        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._thread_loop)
        except Exception as e:
            if self._debug: print(f"send: exception launching coroutine {e}")
        try:
            result = future.result(timeout=3)
        except concurrent.futures.TimeoutError:
            logger.error("send: timeout submitting message to send")
            future.cancel()
            #raise Exception("end: timeout submitting message to send")
            result = False
        except Exception as e:
            if self._debug: print(f'send: unexpected exception waiting for coroutine result {e}')
            raise
        else:
            if self._debug: print(f"send: the coroutine returned {result}")
        finally:
            if self._debug: print(f"send: the coroutine completed")                        

        return result

    # from fedbiomed.common.message import FeedbackMessage, Log
    # m = FeedbackMessage(log=Log(researcher_id='rid', message='test message'))
    def send(self, message: Message):
        """Non-async method for sending messages to researcher 
        
        Method picks over message type provide correct stub RPC call, and
        convert Python dataclass message to gRPC protobuff. 

        """
        # Switch-case for message type and gRPC calls
        match type(message).__name__:
            case FeedbackMessage.__name__:
                return self._run_thread_safe(self._send_from_thread(
                    rpc= self._feedback_stub.Feedback, 
                    message = message.to_proto())
                )
            
            # Rest considered as task reply
            case _:
                return self._run_thread_safe(self._send_from_thread(
                    rpc= self._feedback_stub.ReplyTask, 
                    message = message)
                )

    def start(self):
        """Starts researcher gRPC client"""
        # Runs gRPC async client

        def run():
            try: 
                asyncio.run(
                    #self.connection(), debug=False
                    self.connection(debug=self._debug), debug=False
                )

            # note: needed to catch this exception
            except ClientStop:
                if self._debug:
                    print("Run: caught user stop exception")
            # TODO: never reached, suppress ?
            #
            except Exception as e:
                if self._debug:
                    print(f"Run: caught exception: {e.__class__.__name__}: {e}")
            finally:
                if self._debug:
                    print("Run: finally")


        self._thread = threading.Thread(target=run)
        self._thread.start()
        if self._debug:
            print("start: completed")


    def stop(self):
        """Stop gently running asyncio loop and its thread"""

        if self._thread is None:
            stopped_count = 0
        else:
            stopped_count = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._thread.ident),
                                                                   ctypes.py_object(ClientStop))
        if stopped_count != 1:
            logger.error("stop: could not deliver exception to thread")
        self._thread.join()


    def is_alive(self) -> bool:
        return False if self._thread is None else self._thread.is_alive()


if __name__ == '__main__':

    def handler(signum, frame):
        print(f"Node cancel by signal {signal.Signals(signum).name}")
        rc.stop()
        sys.exit(1)

    rc = ResearcherClient(debug=True)
    signal.signal(signal.SIGHUP, handler)
    rc.start()

    try:
        while rc.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Node cancel by keyboard interrupt")
        try:
            rc.stop()
        except KeyboardInterrupt:
            print("Immediate keyboard interrupt, dont wait to clean")
