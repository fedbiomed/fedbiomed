
import asyncio
import grpc
import threading
import ctypes
import signal


from typing import Callable, Iterable, List
from google.protobuf.message import Message as ProtoBufMessage


from fedbiomed.transport.protocols.researcher_pb2 import Empty
import fedbiomed.transport.protocols.researcher_pb2_grpc as researcher_pb2_grpc


from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, TaskResponse, TaskRequest, FeedbackMessage
from fedbiomed.common.constants import MessageType
from fedbiomed.transport.node_agent import AgentStore, NodeActiveStatus, NodeAgent


import time
import sys


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

    agent_store: AgentStore

    def __init__(
            self, 
            host: str,
            port: str,
            debug: bool = False, 
            on_message: Callable = _default_callback,
    ) -> None:
        """Constructs GrpcServer 

        Args: 
            debug: Activate debug mode for gRPC asyncio
            on_message: Callback function to execute once a message received from the nodes
        """

        self.host = host 
        self.port = port

        self._server = None 
        self._thread = None 
        self._debug = debug
        self._on_message = on_message
        self.loop = None


    async def start(self):
        """Starts gRPC server"""

        self._server = grpc.aio.server( 
            # futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ])

        self.loop = asyncio.get_running_loop()
        self.agent_store = AgentStore(loop=self.loop)

        researcher_pb2_grpc.add_ResearcherServiceServicer_to_server(
            ResearcherServicer(
                agent_store=self.agent_store,
                on_message=self._on_message), 
            server=self._server
        )

        self._server.add_insecure_port(self.host + ':' + str(self.port))

        # Starts async gRPC server
        await self._server.start()

        try:
            if self._debug: 
                logger.debug("Waiting for termination")
            await self._server.wait_for_termination()
        finally:
            if self._debug: 
                logger.debug("Done starting the server")


    async def broadcast(self, message: Message) -> List[NodeAgent]:
        """Broadcasts given message to all active clients.

        Args: 
            message: Message to broadcast

        Returns:
            Node agents that the message broadcasted to. Includes node agents 
                that are in [fedbiomed.transport.node_agent.NodeActiveStatus][NodeActiveStatus.WAITING] 
                status. 
        """

        agents = await self.agent_store.get_all()
        ab = []
        for _, agent in agents.items():
            async with agent.status_lock:
                if agent.status == NodeActiveStatus.DISCONNECTED:
                    logger.info(f"Node {agent.id} is disconnected.")
                    continue

                if agent.status == NodeActiveStatus.WAITING:
                    logger.info(f"Node {agent.id} is in WAITING status. Server is "
                                "waiting for receiving a request from "
                                "this node to convert it as ACTIVE. Node will be updated "
                                "as DISCONNECTED soon if no request received.")

            await agent.send_(message)
            ab.append(agent)      

        return ab

    async def get_agent(self, node_id: str) -> NodeAgent:
        """Gets node agent by given node ID

        Args: 
            node_id: ID of the node whose agent will be retrieved   

        Returns:
            Node agent object to control remote node
        """
        async with self._agent_store_lock:
            return self.agent_store.get(node_id)



class GrpcServer(_GrpcAsyncServer):
    """Grpc server implementation to be used by threads

    This class extends async implementation of gRPC server to be able to
    call async methods from different thread. Currently, it is used by 
    [fedbiomed.researcher.requests.Requests][`Requests`] class that is
    instantiated in the main thread
    """

    def _run(self):
        """Runs asyncio application"""
        try:
            asyncio.run(super().start())
        except Exception as e:
            logger.error(f"Researcher gRPC server has stopped. Please try to restart your kernel, {e}")

    def start(self):
        """Stats async GrpcServer """

        self._thread = threading.Thread(target=self._run, daemon=True)
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

        return self._run_threadsafe(
            super().broadcast(message)
        ) 


    def get_agent(self, node_id: str):
        """Gets node agent by node id
        
        Args:
            node_id: Id of the node
        """
        return self._run_threadsafe(self.agent_store.get(node_id))
    
    def get_all_agents(self):
        """Gets all agents from agent store"""

        return self._run_threadsafe(self.agent_store.get_all())


    def is_alive(self) -> bool:
        """Checks if the thread running gRPC server still alive
        
        Return:
            gRPC server running status
        """
        return False if not isinstance(self._thread, threading.Thread) else self._thread.is_alive()


    def _run_threadsafe(self, coroutine):
        """Runs given coroutine threadsafe
        
        Args:
            coroutine: Awaitable function to be executed as threadsafe
        """

        future = asyncio.run_coroutine_threadsafe(
            coroutine, self.loop
        )

        return future.result()



# TODO: Remove before merging 
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

