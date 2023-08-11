
import asyncio
import grpc
import uuid

from concurrent import futures

import service.message_pb2_grpc as message_pb2_grpc 
import service.message_pb2 as message_pb2

from datetime import datetime


class ResearcherGRPCServer(message_pb2_grpc.FLOrchestratorServicer):

    async def PollExperiment(self, request, context):
        return message_pb2.PollExperimentResponse(experiment_name="Test")

async def serve():
    # initialize server with 4 workers
    server = server = grpc.aio.server()

    # attach servicer method to the server
    message_pb2_grpc.add_FLOrchestratorServicer_to_server(ResearcherGRPCServer(), server)

    # start the server on the port 50051
    server.add_insecure_port("0.0.0.0:50051")
    await server.start()
    print("Started gRPC server: 0.0.0.0:50051")

    # server loop to keep the process running
    await server.wait_for_termination()


# invoke the server method
if __name__ == '__main__':

    asyncio.get_event_loop().run_until_complete(serve())




# class BDStreamServicer(message_pb2_grpc.BDStreamServicer):
#     ''' this servicer method will read the request from the iterator supplied
#         by incoming stream and send back the response in a stream
#     '''
#     def OpenStream(self, request_iterator, context):
#         entry_info = dict()
        
#         print(context)
#         print(dir(context))


#         print(context.peer())
#         # context.send("TEXT")
#         print("#### RPC Peer")
#         print(dir(context.peer))

#         for request in request_iterator:
#             print(request)
#             # stream the response back
#             yield message_pb2.Message(message=f"MEssage {request.message}")

        
