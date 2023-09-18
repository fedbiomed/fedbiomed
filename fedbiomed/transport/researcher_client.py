 

import asyncio
import grpc
import queue
import threading
import sys 
import signal 
import ctypes

from typing import Callable, Dict, Union

from google.protobuf.message import Message as ProtobufMessage

import fedbiomed.transport.protocols.researcher_pb2_grpc as researcher_pb2_grpc

from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, FeedbackMessage, TaskRequest, TaskResult

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
            on_message = None,
            certificate: str = None,
            node_id = None
        ):


        # TODO: implement TLS 
        if certificate is not None:
            # TODO: create channel as secure channel 
            pass 
        
        self._thread = None
        self._client_registered = False
        self.on_message = on_message or (lambda x: print(f"Task received! {x}"))

        self._feedback_channel = create_channel(certificate=None)
        self._feedback_stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._feedback_channel)

        self._task_channel = create_channel(certificate=None)
        self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._task_channel)

        logger.add_grpc_handler(on_log=self._send,
                              node_id=self._messaging_id)

    def connection(self):
        """Create long-lived connection with researcher server"""
        


        logger.info("Waiting for researcher server...")

        # Starts loop to ask for
        while not SHUTDOWN_EVENT.is_set():
             
            try: 
                self.get_tasks()
            except grpc.RpcError as exp:
                if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("Stream TIMEOUT Error")
                    time.sleep(2)
            
                elif exp.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.debug("Researcher server is not available, will retry connect in 2 seconds")
                    time.sleep(2)
                else:
                    raise Exception("Request streaming stopped ") from exp
            except Exception as e:
                raise Exception("Request streaming stopped ") from e

    def get_tasks(self):
        """Lstens for tasks"""
       
        logger.info("Sending new task request")
        while not SHUTDOWN_EVENT.is_set():
            
            self.__request_task_iterator = self._stub.GetTaskUnary(
                TaskRequest(node=f"{NODE_ID}").to_proto()
            )

            # Prepare reply
            reply = bytes()
            for answer in self.__request_task_iterator:
                reply += answer.bytes_
                if answer.size != answer.iteration:
                    continue
                else:
                    # Execute callback
                    self.on_message(Serializer.loads(reply))
                    # Reset reply
                    reply = bytes()


    def _handle_send(
            self, 
            rpc: Callable, 
            proto: Union[ProtobufMessage, Message]
        ) -> grpc:
        """Handle RPC calls
        Args: 
            rpc: 
        Raises: 
            Exception: RPC execution error
        """

        try:
            result = rpc(proto)
            if isinstance(rpc, grpc.UnaryUnaryMultiCallable):
                return rpc(proto)

            elif isinstance(rpc, grpc.StreamUnaryMultiCallable): 
                return rpc(stream_reply(proto))
                    
        except Exception as exp:
            raise Exception("Error while sending message to researcher") from exp
        

    def send(self, message: Message):
        """Send message from node to researcher
        
        Args: 
            message: An instance of Message

        """
        if not isinstance(message, Message):
            raise Exception("The argument message is not fedbiomed.common.message.Message type")


        # Switch-case for message type and gRPC calls
        match type(message).__name__:
            case FeedbackMessage.__name__:
                print("Here")
                self._handle_send(self._feedback_stub.Feedback, message.to_proto())
                
            case _ :
                self._handle_send(self._feedback_stub.ReplyTask, message)
                raise Exception('Undefined message type')

    def start(self):
        """Starts researcher gRPC client"""
        # Create and start background thread

        self._thread = threading.Thread(target=self.connection)
        self._thread.start()

    def stop(self):
        """Stop gently running asyncio loop and its thread"""

        logger.debug("Shutting down researcher client...")
        if self._thread is None:
            stopped_count = 0
        else:
            stopped_count = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._thread.ident),
                                                                   ctypes.py_object(ClientStop))
        if stopped_count != 1:
            logger.error("stop: could not deliver exception to thread")
        self._thread.join()



if __name__ == '__main__':
    
    ResearcherClient().start()