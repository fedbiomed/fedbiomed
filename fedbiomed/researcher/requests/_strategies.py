from typing import List, Dict

from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage
from fedbiomed.transport.node_agent import NodeAgent, NodeActiveStatus



class StrategyStatus:

    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    CONTINUE = "CONTINUE"

class RequestStrategy:
    """Base strategy to collect replies from remote agents"""

    def __init__(self):
        self.nodes = []
        self.replies = {}
        self.nodes_status = {}
        self.status = None 

    def update(
            self, 
            nodes: List[NodeAgent], 
            replies: Dict,
            nodes_status: Dict
    ) -> None:
        """Updates nodes, replies and errors"""

        self.nodes = nodes 
        self.replies = replies
        self.nodes_status = nodes_status


    def continue_(self, requests) -> bool:
        """Default strategy stops collecting result once all nodes has answered
        
        Returns:
            False stops the iteration
        """

        has_finished = all([req.has_finished() for req in requests])
        return self.keep() if not has_finished else self.completed()
        
    def stop(self, status) -> StrategyStatus:
        """Stop sign for strategy"""
        self.status = StrategyStatus.STOPPED
        return StrategyStatus.STOPPED

    def keep(self) -> StrategyStatus:
        """Keeps continue collecting replies from nodes"""
        self.status = StrategyStatus.CONTINUE
        
        return StrategyStatus.CONTINUE

    def completed(self) -> StrategyStatus:
        """Updates status of strategy as completed without any issue"""
        self.status = StrategyStatus.COMPLETED
    
        return StrategyStatus.COMPLETED


class StopOnAnyDisconnect(RequestStrategy):
    """Stops collecting results if a node disconnects"""
    
    def continue_(self, requests) -> bool:
        
        for req in requests:
            if req.status == "DISCONNECT":
                return self.stop(req.status)
        
        return StrategyStatus.CONTINUE


class StopOnAnyError(RequestStrategy):
    """Stops collecting results if a node returns an error"""
    
    def continue_(self, requests):
        
        for req in requests:
            if req.error:
                return self.stop(req.status)
        
        return StrategyStatus.CONTINUE


class StrategyController:

    def __init__(
            self, 
            strategies: List[RequestStrategy], 
            ):
        
        self.strategies = strategies

    def continue_(self, requests) -> bool:
        """Checks if reply collection should continue according to each strategy"""
        status = [strategy.continue_(requests=requests) for strategy in self.strategies]
        status = all(status == StrategyStatus.CONTINUE)

        return StrategyStatus.CONTINUE if status else False
 