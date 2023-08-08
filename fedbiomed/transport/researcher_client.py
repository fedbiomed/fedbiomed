 
import grpc




class ResearcherClient:
    """gRPC researcher component client 
    
    Attributes: 


    """
    def __init__(self, 
                 ip: str, 
                 port: str, 
                 certificate: str = None):


        if certificate is not None:
            # TODO: create channel as secure channel 
            pass 
        self._channel = grpc.aio.insecure_channel()
        self._stub = None # Create stub here
    
    def start():
        pass
        