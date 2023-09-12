import queue
import grpc
import asyncio
import abc

from fedbiomed.proto.researcher_pb2_grpc import ResearcherServiceStub
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, TaskRequest, TaskResult, FeedbackMessage
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH


from typing import List, Callable, Optional, Awaitable

def catch(method):
    """Catches grpc.AioRPCErrors"""

    async def wrapper(*args, **kwargs):

        try: 
            method(*args, **kwargs)

        # Timeout and services un available errors
        except grpc.aio.AioRpcError as exp:
            if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.debug("Timeout ")
        
            elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                await asyncio.sleep(2)

            else:
                raise Exception("get_tasks: Request streaming stopped ") from exp
            
        except Exception as exp:
                raise Exception("get_tasks: Request streaming stopped ") from exp
        
    return wrapper


DEFAULT_ADDRESS = "localhost:50051"


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
        ("grpc.keepalive_time_ms", 1000 * 2),
        ("grpc.initial_reconnect_backoff_ms", 1000),
        ("grpc.min_reconnect_backoff_ms", 500),
        ("grpc.max_reconnect_backoff_ms", 2000),
        # ("grpc.enable_retries", 1), # Not working
        # ("grpc.service_config", service_config) # Not working
    ]

    if certificate is None: 
        channel = grpc.aio.insecure_channel(address, options=channel_options)
    else:
        # TODO: Create secure channel
        pass
    
    # TODO: add callback fro connection state

    return channel



class GrpcClient:
    """An agent of remote researcher gRPC server
    
    Attributes:
        id: Remote researcher ID
        stub: RPC stub use for researcher services
    """
    id: Optional[str]
    task_stub: ResearcherServiceStub
    queue: asyncio.Queue

    
    def __init__(self, ip, host, node_id, loop):
        self._id = None
        self._ip = ip
        self._host = host
        self._node_id = node_id 

        self._feedback_channel = create_channel(certificate=None)
        self.feedback_stub = ResearcherServiceStub(channel=self._feedback_channel)
        
        self._task_channel = create_channel(certificate=None)
        self.task_stub = ResearcherServiceStub(channel=self._task_channel)

        self.task_listener = TaskListener(self.task_stub, self._node_id)
        self.sender = Sender(self.task_stub, self._node_id)


    def start(self, on_task) -> List[Awaitable[asyncio.Task]]:
        """Start researcher gRPC agent.
        
        Starts long-lived task polling and the async queue for the replies
        that is going to be sent back to researcher.

        Args: 
            on_task: Callback function to execute once a task received.
        """
        return [self.task_listener.listen(on_task), self.sender.listen()]


    def send(self, message: Message):
        """Sends messages from node to researcher server"""

        self.sender.send(message)


class Listener:

    def __init__(self, stub: ResearcherServiceStub, node_id: str):
        """Constructs task listener channels
        
        Args: 
            stub: RPC stub to be used for polling tasks from researcher
        """

        self._stub = stub
        self._node_id = node_id

    def listen(self, callback: Optional[Callable] = None) -> Awaitable[asyncio.Task]:
        """Listens for tasks from given channels
        
        Args:
            callback: Callback function 

        Returns:
            Asyncio task to run task listener
        """

        return asyncio.create_task(self._listen(callback))
    
    @abc.abstractmethod
    def _listen(self, callback):
        pass

 
class TaskListener(Listener):
    """Listener for the task assigned by the researcher component """            

    async def _listen(self, callback: Optional[Callable] = None):
        """"Starts the loop for listening task"""

        while True:
            await self._request(callback)


    @catch
    async def _request(self, callback: Optional[Callable] = None) -> None:
        """Requests tasks from Researcher 
        
        Args: 
            researcher: Single researcher GRPC agent
            callback: Callback to execute once a task is arrived
        """
        while True:
            logger.debug(f"Sending new task request to researcher")
            iterator = self._stub.GetTaskUnary(
                TaskRequest(node=f"{self._node_id}").to_proto(), timeout=60
            )
            # Prepare reply
            reply = bytes()
            async for answer in iterator:
                reply += answer.bytes_
                if answer.size != answer.iteration:
                    continue
                else:
                    # Execute callback
                    logger.debug(f"New task received form researcher")
                    if callback:
                        callback(Serializer.loads(reply))
                    
                    # Reset reply
                    reply = bytes()



class Sender(Listener):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = _AsyncQueueBridge()


    async def _listen(self, callback: Optional[Callable] = None):
        """Listens for the messages that are going to be sent to researcher"""

        # While loop retires to send if first one fails to send the result
        while True: 
            # Waits until there is something to send back to researcher
            await self._get(callback)
               

    @catch
    async def _get(self, callback: Optional[Callable] = None):
        """Gets task result from the queue"""
        
        while True:
            msg = await self._send_queue.get()

            # If it is aUnary-Unary RPC call
            if isinstance(msg["stub"], grpc.aio.UnaryUnaryMultiCallable):
                await msg["stub"](msg["message"])

            elif isinstance(msg["stub"], grpc.aio.grpc.aio.StreamUnaryMultiCallable): 
                
                stream_call = msg["stub"]()
                
                if callback:
                    callback(msg["message"])

                for reply in self._stream_reply(msg["message"]):
                    await stream_call.write(reply)

                await stream_call.done_writing()

            self._send_queue.task_done()


    def _stream_reply(message: Message):
        """Streams task result back researcher component"""

        reply = Serializer.dumps(message.get_dict())
        chunk_range = range(0, len(reply), MAX_MESSAGE_BYTES_LENGTH)
        for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
            stop = start + MAX_MESSAGE_BYTES_LENGTH 
            yield TaskResult(
                size=len(chunk_range),
                iteration=iter_,
                bytes_=reply[start:stop]
            ).to_proto()


    def send(self, message: Message):

        # TODO: refactor !
        # self._node_configured self._task_channel and self._feedback_stub 
        # should be used only in spawn thread, not in master thread
        if not self.is_connected():
            raise Exception("send: the connection is not ready")
        
        # Switch-case for message type and gRPC calls
        match message.__class__.__name__:
            case FeedbackMessage.__name__:
                # Note: FeedbackMessage is designed as proto serializable message.
                self._queue.put_threadsafe({"stub": self._stub.Feedback, "message": message.to_proto()})

                        
            case _:
                # TODO: All other type of messages are not strictly typed
                # on gRPC communication layer. Those messages are going to be 
                # send as TaskResult which are formatted as bytes of data.
                # The future development should type every message on GRPC layer
                self._queue.put_threadsafe({"stub": self._stub.Feedback, "message": message}) 
                    
                

class _AsyncQueueBridge(asyncio.Queue): 
    """Provides threadsafe operations for put and get """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loop_ = asyncio.get_running_loop()

    def put_threadsafe(self, item):
        """Executes put_nowait as thread safe 
        
        Args: 
            item: Item to put in the queue
        """
        self.loop_.call_soon_thread_safe(
            super().put_nowait, item
        )

    def get_threadsafe(self):
        """Executes get_nowait threadsafe """
        return self.loop_.call_soon_thread_safe(
            super().get_nowait,
        )
    
