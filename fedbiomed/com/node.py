import asyncio
import logging
import random
from typing import Iterable, List

import grpc
import service.message_pb2_grpc as message_pb2_grpc 
import service.message_pb2 as message_pb2



class Node:

    def __init__(self):

        self.researcher_channel = grpc.aio.insecure_channel('localhost:50051')
        self.stub = message_pb2_grpc.FLOrchestratorStub(channel=self.researcher_channel)

    def run(self):

        while True:     
            with self.researcher_channel:
                experiment = self.stub.PollExperiment(node="Dummy node")
                print(experiment)

if __name__ == '__main__':
    node = Node()
    node.run()

# if __name__ == '__main__':
#     asyncio.get_event_loop().run_until_complete(main())


# def request_iterator():
#     for idx in range(1, 16):
#         entry_request = message_pb2.Message(message=f"Test message {idx}")
#         yield entry_request


# # iterate through response stream and print to console
# async def send_stream(stub: message_pb2_grpc.BDStreamStub):
#     async for entry_response in stub.OpenStream(request_iterator()):
#         print(entry_response.message)

# async def main() -> None:
#     # create service stub
#     async with grpc.aio.insecure_channel('localhost:50051') as channel:
#         stub = message_pb2_grpc.BDStreamStub(channel=channel)
#         await send_stream(stub)


