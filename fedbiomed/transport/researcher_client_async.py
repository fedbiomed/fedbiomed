 

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

import fedbiomed.transport.protocols.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.transport.protocols.researcher_pb2 import TaskRequest, FeedbackMessage
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH
from fedbiomed.common.message import Message, TaskRequest, FeedbackMessage, TaskResult, ProtoSerializableMessage

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


# Method configuration for retry polic
# NOTE: DIDN'T WORK
# See: https://github.com/grpc/proposal/blob/master/A6-client-retries.md#retry-policy
# and https://fuchsia.googlesource.com/third_party/grpc/+/HEAD/doc/service_config.md
# The initial retry attempt will occur at random(0, initialBackoff). 
# In general, the n-th attempt will occur at random(0, min(initialBackoff*backoffMultiplier**(n-1), maxBackoff)).





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


def stream_reply(message: Message):

    reply = Serializer.dumps(message.get_dict())
    chunk_range = range(0, len(reply), MAX_MESSAGE_BYTES_LENGTH)
    for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
        stop = start + MAX_MESSAGE_BYTES_LENGTH 
        yield TaskResult(
            size=len(chunk_range),
            iteration=iter_,
            bytes_=reply[start:stop]
        ).to_proto()
        

class ResearcherClient:
    """gRPC researcher component client 
    
    Attributes: 


    """
    def __init__(
            self,
            node_id: str = NODE_ID,
            on_message: Callable = None,
            certificate: str = None,
            debug: bool = False
        ):

        self._node_id = node_id
        # TODO: implement TLS 
        if certificate is not None:
            # TODO: create channel as secure channel 
            pass 
        
        self._client_registered = False
        self.on_message = on_message or self._default_callback

        self._thread = None
        self._task_channel = None
        self._feedback_channel = None
        # note: _node_configured is not enough, many race conditions remaining ...
        self._node_configured = False

        self._debug = debug

        logger.add_grpc_handler(on_log=self.send,
                              node_id=self._node_id)
        

    def _default_callback(self, x):
        print(f'Default callback: {type(x)} {x}')


    async def connection(self, debug: bool = False):
        """Create long-lived connection with researcher server"""
        
        self._feedback_channel = create_channel(certificate=None)
        self._feedback_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)
        
        self._task_channel = create_channel(certificate=None)
        self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)

        # need to recreate a queue for each thread to avoid a 
        # unexpected exception <class 'RuntimeError'> <Queue at 0x12345678 maxsize=0> is bound to a different event loop
        self._send_queue = asyncio.Queue()
        self._thread_loop = asyncio.get_running_loop()
        self._node_configured = True

        logger.info("Waiting for researcher server...")

        #await asyncio.gather(
        #    self.get_tasks(debug),
        #    self.send_queue_listener(debug),
        #)

        task_get = None
        task_send = None
        try:

            task_get = asyncio.create_task(self.get_tasks(debug=debug))
            task_send = asyncio.create_task(self.send_queue_listener(debug=debug))

            await asyncio.wait([task_get, task_send])

            # never reach this one normally ?
            if debug: print('connection: tasks completed')
        except Exception as e:
            if debug:print(f"connection: unexpected exception {type(e)} {e}")
        finally:
            self._node_configured = False
            if debug: print("connection: finally")

    async def send_queue_listener(self, debug: bool = False):
        """Listens queue that contains message to send to researcher """


        while True:

            try:
                while True: 
                    msg = await self._send_queue.get()

                    # If it is aUnary-Unary RPC call
                    if isinstance(msg["stub"], grpc.aio.UnaryUnaryMultiCallable):
                        logger.debug("In feedback!")
                        await msg["stub"](msg["message"])

                    elif isinstance(msg["stub"], grpc.aio.grpc.aio.StreamUnaryMultiCallable): 
                        logger.debug("Here in StreamUnaryMulltipllable")
                        stream_call = msg["stub"]()

                        for reply in stream_reply(msg["message"]):
                            await stream_call.write(reply)

                        await stream_call.done_writing()

                    self._send_queue.task_done()

            except grpc.aio.AioRpcError as exp:
                if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("send_queue_listener: Stream TIMEOUT Error")
                    print(exp)
                    await asyncio.sleep(1)

                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"send_queue_listener: {self._task_channel.get_state()}")

                    logger.debug("send_queue_listener: Researcher server is not available, will retry connect in 2 seconds")
                    await asyncio.sleep(2)
                else:
                    if debug:
                        print(f'send_queue_listener: unknown exception {exp.__class__.__name__} {exp}')
                    raise Exception("send_queue_listener: Request streaming stopped ") from exp
            except Exception as e:
                print(e)
                if debug:
                    print(f"send_queue_listener: unexpected exception {type(e)} {e}")
            finally:
                # Cancel the call
                # stream_call.cancel()
                # self._send_queue.task_done()
                if debug:
                    print("send_queue_listener: finally")


    async def get_tasks(self, debug: bool = False):
        """Long-lived polling to request tasks from researcher."""

        while True:

            try:
                # await task_reader(stub= self._stub, node=NODE_ID, callback=self.on_message)
                # await task_reader_unary(stub= self._stub, node=NODE_ID, callback= lambda x: x)
                while True:
                    logger.info("Sending new task request")
                    self.__request_task_iterator = self._stub.GetTaskUnary(
                        TaskRequest(node=f"{self._node_id}").to_proto(), timeout=60
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
                    logger.debug("get_tasks: Stream TIMEOUT Error")
                    print(exp)
            
                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"get_tasks: {self._task_channel.get_state()}")
                    logger.debug("get_tasks: Researcher server is not available, will retry connect in 2 seconds")
                    
                    await asyncio.sleep(2)
                else:
                    if debug: print(f'get_tasks: unknown exception {exp.__class__.__name__} {exp}')
                    raise Exception("get_tasks: Request streaming stopped ") from exp
            except Exception as e:
                if debug:
                    print(f"get_tasks: unexpected exception {type(e)} {e}")
            finally:
                if debug: print('get_tasks: finally')
          


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
        
        if not isinstance(message, (ProtobufMessage, Message)):
            raise Exception("'message' should be be an instance of Protobuf or Message")

        return {"stub": stub, "message": message}
    
    #async def _send_from_thread(self, rpc, message: ProtobufMessage):
    #    try:
    #        #await asyncio.sleep(5)
    #        print("About to add task in the queue")
    #        await self._send_queue.put(
    #            self._create_send_task(rpc,message)
    #        )
    #        print("task added in the queue")
    #        return True
    #    except asyncio.CancelledError:
    #        if self._debug: print("_send_from_thread: cancelled send task")
    #        return False


    #def _run_thread_safe(self, coroutine: asyncio.coroutine):
    #    """Runs given coroutine as thread-safe"""
#
    #    try:
    #        future = asyncio.run_coroutine_threadsafe(coroutine, self._thread_loop)
    #    except Exception as e:
    #        if self._debug: print(f"send: exception launching coroutine {e}")
#
    #    try:
    #        result = future.result(timeout=20)
    #    except concurrent.futures.TimeoutError:
    #        logger.info("send: timeout submitting message to send")
    #        future.cancel()
    #        #raise Exception("end: timeout submitting message to send")
    #        result = False
    #    except Exception as e:
    #        if self._debug: print(f'send: unexpected exception waiting for coroutine result {e}')
    #        raise
    #    else:
    #        if self._debug: print(f"send: the coroutine returned {result}")
    #    finally:
    #        if self._debug: print(f"send: the coroutine completed")                        
#
    #    return result

    # from fedbiomed.common.message import FeedbackMessage, Log
    # m = FeedbackMessage(log=Log(researcher_id='rid', message='test message'))
    def send(self, message: Message):
        """Non-async method for sending messages to researcher 
        
        Method picks over message type provide correct stub RPC call, and
        convert Python dataclass message to gRPC protobuff. 

        """
        # TODO: refactor !
        # self._node_configured self._task_channel and self._feedback_stub 
        # should be used only in spawn thread, not in master thread
        if not self.is_connected():
            raise Exception("send: the connection is not ready")
        
        # Switch-case for message type and gRPC calls
        match message.__class__.__name__:
            case FeedbackMessage.__name__:
                # Note: FeedbackMessage is designed as proto serializable message.
                self._thread_loop.call_soon_threadsafe(
                    self._send_queue.put_nowait, 
                    self._create_send_task(self._stub.Feedback, message.to_proto()))        
            case _:
                # TODO: All other type of messages are not strictly typed
                # on gRPC communication layer. Those messages are going to be 
                # send as TaskResult which are formatted as bytes of data.
                # The future development should type every message on GRPC layer
                self._thread_loop.call_soon_threadsafe(
                    self._send_queue.put_nowait, 
                    self._create_send_task(self._stub.ReplyTask, message))


    def start(self):
        """Starts researcher gRPC client"""
        # Runs gRPC async client

        if self._node_configured:
            logger.info("start: node client is already started")
            return

        def run():
            try: 
                asyncio.run(
                    #self.connection(), debug=False
                    self.connection(debug=self._debug), debug=True
                )
            except Exception as e:
                if self._debug:
                    print(f"Run: caught exception: {e.__class__.__name__}: {e}")
            finally:
                if self._debug:
                    print("Run: finally")


        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        if self._debug:
            print("start: completed")

    def is_alive(self) -> bool:
        return False if not isinstance(self._thread, threading.Thread) else self._thread.is_alive()

    def is_connected(self) -> bool:
        """Node is configured and ready for communication
        """
        # TODO: refactor need more checks
        return self.is_alive() and self._node_configured and isinstance(self._task_channel, grpc.aio.Channel) and \
            self._task_channel.get_state() == grpc.ChannelConnectivity.READY

if __name__ == '__main__':

    def handler(signum, frame):
        print(f"Node cancel by signal {signal.Signals(signum).name}")
        sys.exit(1)

    rc = ResearcherClient(debug=True)
    signal.signal(signal.SIGHUP, handler)
    rc.start()

    try:
        while rc.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Node cancel by keyboard interrupt")
