
import asyncio
import grpc

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.proto.researcher_pb2 import RegisterResponse, GetTaskResponse
from fedbiomed.common.logger import logger
from concurrent import futures
import time

DEFAULT_PORT = 50051
DEFAULT_HOST = 'localhost'

logger.setLevel("INFO")
class ResearcherServicer(researcher_pb2_grpc.ResearcherServiceServicer):

    def Register(self, request, unused_context):

        node_id = request.node 
        logger.info("Node register request from {node_id}")
        print(node_id)
        return RegisterResponse(message=f"registration completed {node_id}", status=True)

    async def GetTask(self, request_iterator, context):

        async for req in request_iterator:
            
            print("Request has arrived")
            print(req)
            node = req.node
            logger.info(f" Node : {node} requester a task")
            await asyncio.sleep(1)
            yield GetTaskResponse(task_id=f"test-task-{0}", context="Test context")

class ResearcherServer:

    def __init__(self) -> None:

        self._server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=1))
        researcher_pb2_grpc.add_ResearcherServiceServicer_to_server(
            ResearcherServicer(), 
            server=self._server
            )
    

    async def start(self):

        self._server.add_insecure_port(DEFAULT_HOST + ':' + str(DEFAULT_PORT))

        logger.info("Starting researcher service...")
        await self._server.start()
        await self._server.wait_for_termination()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(ResearcherServer().start())
