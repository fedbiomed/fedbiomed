 

import asyncio
import grpc
import queue
import threading
import sys 
import signal 

from typing import Callable, Dict

from google.protobuf.message import Message as ProtobufMessage

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc

from fedbiomed.proto.researcher_pb2 import TaskRequest, FeedbackMessage, Log
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, Scalar, LogMessage

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


SHUTDOWN_EVENT =  threading.Event()



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
        channel = grpc.insecure_channel(address, options=channel_options)
    else:
        # TODO: Create secure channel
        pass
    
    # TODO: add callback fro connection state

    return channel


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
        self.on_message = lambda x: x

        self._feedback_channel = create_channel(certificate=None)
        self._feedback_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)

        self._task_channel = create_channel(certificate=None)
        self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)

    def connection(self):
        """Create long-lived connection with researcher server"""
        


        logger.info("Waiting for researcher server...")

        # Starts loop to ask for
        self.get_tasks()   
        

    def get_tasks(self):

        while not SHUTDOWN_EVENT.is_set():
            logger.info("Sending new task request")
            try:
                # await task_reader(stub= self._stub, node=NODE_ID, callback=self.on_task)
                # await task_reader_unary(stub= self._stub, node=NODE_ID, callback= lambda x: x)
                while not SHUTDOWN_EVENT.is_set():
                    
                    self.__request_task_iterator = self._stub.GetTaskUnary(
                        TaskRequest(node=f"{NODE_ID}")
                    )

                    # Prepare reply
                    reply = bytes()
                    for answer in self.__request_task_iterator:
                        print('----')
                        reply += answer.bytes_
                        if answer.size != answer.iteration:
                            continue
                        else:
                            # Execute callback
                            self.on_message(Serializer.loads(reply))
                            # Reset reply
                            reply = bytes()
            
            except grpc.aio.AioRpcError as exp:
                print(exp.code())
                if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("Stream TIMEOUT Error")
                    time.sleep(2)
            
                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                    time.sleep(2)
                else:
                    raise Exception("Request streaming stopped ") from exp
            finally:
                pass
            
    
    def _handle_send(
            self, 
            rpc: Callable, 
            proto: ProtobufMessage
        ) -> grpc:
        """Handle RPC calls
        Args: 
            rpc: 
        Raises: 
            Exception: RPC execution error
        """

        try:
            result = rpc(proto)
        except:
            raise Exception("Error while sennding message to researcher")
        
        return result 

    def send(self, message: Message):
        """Send message from node to researcher
        
        Args: 
            message: An instance of Message

        """

        if not isinstance(message, Message):
            raise Exception("The argument message is not fedbiomed.common.message.Message type")


        # Switch-case for message type and gRPC calls
        match message.__name__:
            case Scalar.__name__ | LogMessage.__name__:
                self._handle_send(self._feedback_stub.Feedback, message.to_proto())
                
            case _ :
                raise Exception('Undefined message type')

        # Convert message to proto
        proto = message.to_proto()


        try:
            self._feedback_stub.Feedback(
                    Log(log=log)
                )
        except Exception as e:
            print(e)
            pass
        

    def start(self):
        """Starts researcher gRPC client"""
        # Runs gRPC async client

        if SHUTDOWN_EVENT.is_set(): 
            SHUTDOWN_EVENT.clear()


        # Create and start background thread
        t = threading.Thread(target=self.connection)
        t.start()


    def stop(self):
        """Stop gently running asyncio loop and its thread"""

        logger.debug("Shutting down researcher client...")
        self.__request_task_iterator.cancel()
        SHUTDOWN_EVENT.set()

if __name__ == '__main__':
    
    ResearcherClient().start()