
import asyncio
import grpc

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc

from fedbiomed.proto.researcher_pb2 import TaskResponse, TaskResponseUnary
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from concurrent import futures
import time
import random 
import json 
import sys

DEFAULT_PORT = 50051
DEFAULT_HOST = 'localhost'

logger.setLevel("INFO")

rnd = random.sample(range(0, 1000000), 100000)

training_task = {
    "researcher_id" : "test-researcher",
    "job_id" : "test-job",
     "training_args" : {"args_1": 12, "args_2": True, "args_3": "string"},
     "model_args": {"args_1": 12, "args_2": True, "args_3": "string"},
    "dataset_id": "ddd-id",
    "training": True, 
    "training_plan": "Basic training plan",
    "secagg_servkey_id": None,
    "secagg_biprime_id ": None,
    "secagg_random": None, 
    "secagg_clipping_range":  None,
    "round": 1,
    "aggregator_args": {"args_1": 12, "args_2": True, "args_3": "string"},
    "aux_vars": {"arg_1": rnd, "args_2": rnd, "args_3": rnd, "args_4": rnd, "args_5": rnd, "args_6": rnd, "args_7": rnd,
                 "arg_1x": rnd, "args_2x": rnd, "args_3x": rnd, "args_4x": rnd, "args_5x": rnd, "args_6x": rnd, "args_7x": rnd,
                 "arg_1y": rnd, "args_2y": rnd, "args_3y": rnd, "args_4y": rnd, "args_5y": rnd, "args_6y": rnd, "args_7y": rnd,
                 "arg_1z": rnd, "args_2z": rnd, "args_3z": rnd, "args_4z": rnd, "args_5z": rnd, "args_6z": rnd, "args_7z": rnd,
                 "arg_1d": rnd, "args_2d": rnd, "args_3d": rnd, "args_4d": rnd, "args_5d": rnd, "args_6d": rnd, "args_7d": rnd
                  },
    "model_params": {"arg_1": rnd, "args_2": rnd, "args_3": rnd, "args_4": rnd, "args_5": rnd, "args_6": rnd, "args_7": rnd },

}


small_task = {
        "researcher_id" : "test-researcher",
    "job_id" : "test-job",
     "training_args" : {"args_1": 12, "args_2": True, "args_3": "string"},
     "model_args": {"args_1": 12, "args_2": True, "args_3": "string"},
    "dataset_id": "ddd-id",
    "training": True, 
    "training_plan": "Basic training plan",
    "secagg_servkey_id": None,
    "secagg_biprime_id ": None,
    "secagg_random": None, 
    "secagg_clipping_range":  None,
    "round": 1,
}


# Max message length as bytes
MAX_MESSAGE_BYTES_LENGTH = 4000000 - sys.getsizeof(bytes("", encoding="UTF-8")) # 4MB 

class ResearcherServicer(researcher_pb2_grpc.ResearcherServiceServicer):


    async def GetTask(self, request_iterator, context):
        

        
        async for req in request_iterator:
            
            # print("Request has arrived")
            # print(req)
            node = req.node

            # Here call task and wait until there is no

            logger.info(f"Received request form {node}")
            # Dumps the message
            task = Serializer.dumps(small_task)
            chunk_range = range(0, len(task), MAX_MESSAGE_BYTES_LENGTH)
            for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
                stop = start + MAX_MESSAGE_BYTES_LENGTH 
                yield TaskResponse(
                    size=len(chunk_range),
                    iteration=iter_,
                    bytes_=task[start:stop]
                )

            logger.info(f"Request finished for {node}")


    def GetTaskUnary(self, request, context):

        node = request.node

        # Here call task and wait until there is no

        logger.info(f"Received request form {node}")
        task = Serializer.dumps(small_task)
        chunk_range = range(0, len(task), MAX_MESSAGE_BYTES_LENGTH)
        time.sleep(10)
        for start, iter_ in zip(chunk_range, range(1, len(chunk_range)+1)):
            stop = start + MAX_MESSAGE_BYTES_LENGTH 
            yield TaskResponse(
                size=len(chunk_range),
                iteration=iter_,
                bytes_=task[start:stop]
            )

class ResearcherServer:

    def __init__(self) -> None:
        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10), 
            options=[
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
            )
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
