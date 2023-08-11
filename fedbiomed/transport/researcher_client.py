 

import asyncio
import grpc
import queue

from typing import Callable

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.proto.researcher_pb2 import RegisterRequest, GetTaskRequest
from fedbiomed.common.logger import logger

import uuid
import time



class GRPCTimeOUT(Exception):
    """gRPC timeout error"""
    pass 


class GRPCStreamingKeepAliveExceed(Exception):
    """"Task reader keep alive error"""
    pass

logger.setLevel("DEBUG")
NODE_ID = str(uuid.uuid4())


tas_queue = queue.Queue()

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

    if certificate is None: 
        channel = grpc.aio.insecure_channel(address)
    else:
        # TODO: Create secure channel
        pass
    
    # TODO: add callback fro connection state

    return channel


async def task_reader(stub, node: str, callback: Callable = None):
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
            GRPCTimeOut: Once the timeout is exceeded reader raises time out exception and 
                closes the streaming.
        """  
        #condition = asyncio.Condition()
        event = asyncio.Event()

        async def request_tasks(event):
            """Send request stream to researcher server"""
            async def request_():
                # Send starting request without relying on a condition
                count  = 1
                logger.info("Sending first request after creating a new stream connection")
                yield GetTaskRequest(node=f"{count}---------{node}")
                
                count = 2
                timeout = time.time() + 60 * 5 
                # Cycle of requesting for tasks
                
                while True:
                    # Won't be executed until condition is notified by the
                    # task that listens for server responses
                    logger.info("It came here")

                    print("Acquire condition")
                    # await condition.acquire()
                    # # async with condition:
                    # try:
                    #     print("Wait condition")
                    #     await condition.wait()
                    # finally: 
                    #     print("Release condition")
                    #     condition.release()
                    await event.wait()
                    event.clear()
                    if time.time() > timeout:
                        raise GRPCStreamingKeepAliveExceed() 

                    #logger.info(f"Sending another request ---- within the stream")     
                    yield GetTaskRequest(node=f"{count}---------{node}")
                    count += 1
                    #


            # Call request iterator withing GetTask stub
            state.task_iterator = stub.GetTask(
                        request_()
                    )

        async def receive_tasks(event):
            """Receives tasks form researcher server """

            async for answer in state.task_iterator:
                print("Received response")
                print(answer)
            
                # async with condition:
                #     condition.notify()
                #     print("Notified")
                
                event.set()

                # try:
                #     task = tas_queue.get(block=False)
                # except Exception:
                #     continue

        # Shared state between two coroutines
        state = type('', (), {})()

        try:
            await asyncio.gather(
                request_tasks(event), 
                receive_tasks(event)
                )
        except grpc.aio.AioRpcError as exp:
            if exp.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.debug("Stream TIMEOUT Error")
                raise GRPCTimeOUT(f"Timeout exceed for streaming") from exp
            else:
                print(exp)
                raise Exception("Request streaming stopped ") from exp
        finally:
            pass




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
        self.handler = handler

    async def connection(self):
        """Create long-lived connection with researcher server"""
        
        async with create_channel(certificate=None) as channel:
            
            self._channel = channel
            self._stub = researcher_pb2_grpc.ResearcherServiceStub(channel=self._channel)

            logger.info("Waiting for researcher server...")
            #task = asyncio.create_task(self._register())

            # Node first sends registration request to researcher
            await self._register()

            # Starts loop to ask for
            # asyncio.ensure_future(self.get_tasks())
            await self.get_tasks()   
            


    async def _register(self):
        """Register loop for researcher client
        
        """
        while True: 
            print("Here")
            try:
                response = await self._stub.Register(
                    RegisterRequest(node=NODE_ID)
                )
                print(response)
                if response.status == True:
                    self._client_registered = True
                    logger.info("Client has been registered on researcher server ")
                    break
                
            except grpc.aio.AioRpcError as exp: 
                
                if exp.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.debug("gRPC service is not available...")
                    time.sleep(2)
                    continue
                else:
                    logger.error("Something went wrong!")
                    break
        return 
    

   
    async def get_tasks(self, callback = None):

        while True:
            
            if self._client_registered:
                try:

                    logger.info("Sending new task request")
                    try:
                        await task_reader(stub= self._stub, node=NODE_ID)
                    except GRPCTimeOUT:
                        # Continue by creating a new streaming connection
                        continue
                    except GRPCStreamingKeepAliveExceed as exp:
                        continue

                except Exception as e:
                    # Retry request if server is down
                    print(e)
                    time.sleep(2)
            time.sleep(2)

    def start(self):

        try: 
           asyncio.run(
                self.connection(), debug=True
            )
        except KeyboardInterrupt:
            asyncio.get_running_loop().close()

        
    

if __name__ == '__main__':
    ResearcherClient().start()