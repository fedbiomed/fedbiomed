from typing import List, Callable, Optional, Awaitable
from enum import Enum
import asyncio
import abc

import grpc

from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub

from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, TaskRequest, TaskResult, FeedbackMessage
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedCommunicationError


class ClientStatus(Enum):
    DISCONNECTED = 0
    CONNECTED = 1


def create_channel(
    port: str,
    host: str,
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
        ("grpc.max_reconnect_backoff_ms", 2000)
    ]

    if certificate is None: 
        channel = grpc.aio.insecure_channel(f"{host}:{port}", options=channel_options)
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


    def __init__(self, node_id: str, researcher, update_id_map: Callable):

        self._id = None

        self._port = researcher.port
        self._host = researcher.host
        self._node_id = node_id 

        self._feedback_channel = create_channel(port=researcher.port, host=researcher.host, certificate=None)
        self.feedback_stub = ResearcherServiceStub(channel=self._feedback_channel)

        self._task_channel = create_channel(port=researcher.port, host=researcher.host, certificate=None)
        self.task_stub = ResearcherServiceStub(channel=self._task_channel)

        self.task_listener = TaskListener(
            stub=self.task_stub, 
            node_id=self._node_id, 
            on_status_change = self._on_status_change, 
            update_id=self._update_id)

        self.sender = Sender(
            feedback_stub=self.feedback_stub, 
            task_stub=self.task_stub,
            node_id=self._node_id)

        self._loop = asyncio.get_running_loop()
        self._running_tasks = []
        self._status  = ClientStatus.DISCONNECTED
        self._update_id_map = update_id_map
        self.tasks = []

    def start(self, on_task) -> List[Awaitable[asyncio.Task]]:
        """Start researcher gRPC agent.

        Starts long-lived task polling and the async queue for the replies
        that is going to be sent back to researcher.

        Args: 
            on_task: Callback function to execute once a task received.
        """
        self.tasks = [self.task_listener.listen(on_task), self.sender.listen()]

        return self.tasks


    async def send(self, message: Message):
        """Sends messages from node to researcher server"""

        return await self.sender.send(message)


    def _on_status_change(self, status: ClientStatus):
        """Callback function to call once researcher status is changed

        Args: 
            status: New status of the researcher client
        """
        self._status = status

    async def _update_id(self, id_: str):
        """Updates researcher ID

        Args: 
            id_: Researcher Id 
        """
        self._id = id_
        await self._update_id_map(f"{self._host}:{self._port}", id_)


class Listener:

    def __init__(
            self, 
            node_id: str, 
    ) -> None:
        """Constructs task listener channels

        Args: 
            stub: RPC stub to be used for polling tasks from researcher
        """

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

    def __init__(
            self,  
            node_id: str, 
            stub,
            on_status_change: Optional[Callable] = None,
            update_id: Optional[Callable] = None
    ) -> None:

        super().__init__(node_id=node_id)

        self._stub = stub
        self._on_status_change = on_status_change
        self._update_id = update_id


    async def _listen(self, callback: Optional[Callable] = None):
        """"Starts the loop for listening task

        Args: 
            callback: Callback to execute once a task is received
        """

        while True:

            try:
                await self._request(callback)
            except grpc.aio.AioRpcError as exp:
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        logger.debug(
                            f"TaskListener has reach timeout. Re-sending request to {'researcher'} collect tasks")

                    case grpc.StatusCode.UNAVAILABLE:
                        logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                        self._on_status_change(ClientStatus.DISCONNECTED)
                        await asyncio.sleep(2)
                    case grpc.StatusCode.UNKNOWN:
                        logger.debug("Unexpected error raised by researcher gRPC server. This is probably due to "
                                     f"bug on the researcher side. {exp}")
                        raise FedbiomedCommunicationError(
                            f"{ErrorNumbers.FB628}: Task listener stopped " "due to error on the researcher side")
                    case _:
                        raise FedbiomedCommunicationError(
                            f"{ErrorNumbers.FB628}: Unhandled gRPC call status {exp.code()}. Exception: {exp}") from exp

            except Exception as exp:
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Task listener has stopped due to unknown reason: {exp}") from exp


    async def _request(self, callback: Optional[Callable] = None) -> None:
        """Requests tasks from Researcher 

        Args: 
            researcher: Single researcher GRPC agent
            callback: Callback to execute once a task is arrived
        """
        while True:
            logger.debug("Sending new task request to researcher")
            self._on_status_change(ClientStatus.CONNECTED)
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
                    logger.debug("New task received form researcher")
                    task = Serializer.loads(reply)

                    await self._update_id(task["researcher_id"])

                    if callback:
                        callback(task)

                    # Reset reply
                    reply = bytes()
            # Update status as connected


class Sender(Listener):

    def __init__(
            self,  
            node_id: str, 
            feedback_stub: ResearcherServiceStub,
            task_stub: ResearcherServiceStub,
    ) -> None:

        super().__init__(node_id=node_id)
        self._queue = _AsyncQueueBridge()
        self._task_stub = task_stub
        self._feedback_stub = feedback_stub
        self._retry_count = 0

    async def _listen(self, callback: Optional[Callable] = None):
        """Listens for the messages that are going to be sent to researcher"""

        # While loop retires to send if first one fails to send the result
        while True: 

            # Waits until there is something to send back to researcher
            try:
                await self._get(callback)
            except grpc.aio.AioRpcError as exp:
                if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("Timeout reached. Researcher might be busy. ")
                    self._queue.task_done()
                    pass
                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.debug("Researcher server is not available, will try to to send the message in 5 seconds")
                    self._retry_count += 1
                    await asyncio.sleep(3)


    async def _get(self, callback: Optional[Callable] = None):
        """Gets task result from the queue"""

        while True:
            if self._retry_count > 5:
                logger.warning("Message can not be sent to researcher after 5 retries")
                self._queue.task_done()

            msg = await self._queue.get()
            # If it is aUnary-Unary RPC call
            if isinstance(msg["stub"], grpc.aio.UnaryUnaryMultiCallable):
                await msg["stub"](msg["message"])

            elif isinstance(msg["stub"], grpc.aio.StreamUnaryMultiCallable): 
                stream_call = msg["stub"]()

                if callback:
                    callback(msg["message"])

                for reply in self._stream_reply(msg["message"]):
                    await stream_call.write(reply)

                await stream_call.done_writing()

            self._queue.task_done()
            self._retry_count = 0

    def _stream_reply(self, message: Message):
        """Streams task result back researcher component"""

        reply = Serializer.dumps(message.get_dict())
        chunk_range = range(0, len(reply), MAX_MESSAGE_BYTES_LENGTH)
        for start, iter_ in zip(chunk_range, range(1, len(chunk_range) + 1)):
            stop = start + MAX_MESSAGE_BYTES_LENGTH 
            yield TaskResult(
                size=len(chunk_range),
                iteration=iter_,
                bytes_=reply[start:stop]
            ).to_proto()


    async def send(self, message: Message):
        # Switch-case for message type and gRPC calls
        match message.__class__.__name__:
            case FeedbackMessage.__name__:
                # Note: FeedbackMessage is designed as proto serializable message.
                return await self._queue.put({"stub": self._feedback_stub.Feedback, "message": message.to_proto()})

            case _:
                return await self._queue.put({"stub": self._task_stub.ReplyTask, "message": message}) 



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
        self.loop_.call_soon_threadsafe(
            super().put_nowait, item
        )

    def get_threadsafe(self):
        """Executes get_nowait threadsafe """
        return self.loop_.call_soon_threadsafe(
            super().get_nowait,
        )
