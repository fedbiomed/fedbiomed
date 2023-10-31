from typing import List

from fedbiomed.common.logger import logger
from fedbiomed.transport.node_agent import NodeAgent, NodeActiveStatus



class StrategyStatus:

    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    UNSET = "UNSET"

class RequestStrategy:
    """Base strategy to collect replies from remote agents"""

    def __init__(self):
        self.nodes = {}
        self.replies = {}
        self.errors = {}
        self.nodes_status = {}
        self.status = None 


    def update(self, nodes: List[NodeAgent], replies, errors):
        """Updates nodes, replies and errors"""

        self.nodes = nodes 
        self.replies = replies 
        self.errors = errors
        
        for node in nodes:
            self.nodes_status.update({
                node.id: {
                    'status': node.status,
                    'error': True if errors.get(node.id) else False,
                    'reply': True if replies.get(node.id) else False
                }
                 
            })

    def continue_(self) -> bool:
        """Default strategy stops collecting result once all nodes has answered
        
        Returns:
            False stops the iteration
        """
        
        return not all(self.has_answered(node.id) for node in self.nodes)
        
    
    def has_answered(self, node_id):
        """Check if node has any answer"""
        return self.has_error(node_id) or self.has_reply(node_id)


    def has_error(self, node_id):
        """Check if node has replied with error"""
        return self.nodes_status[node_id]['error']


    def has_reply(self, node_id):
        """Check if node has replied with non error"""
        return self.nodes_status[node_id]['reply']

    def has_not_answered_yet(self):

        return [node for node in self.nodes if not self.has_answered(node.id)]

class StrategyController:

    def __init__(self, strategies: List[RequestStrategy]):
        self.strategies = strategies

    def continue_(self) -> bool:
        """Checks if reply collection should continue according to each strategy"""
        status = [strategy.continue_() for strategy in self.strategies]
        
        return all(status)

    def update(self, nodes, replies, errors):
        """Updates node, replies and error states of each strategy"""
        for strategy in self.strategies:
            strategy.update(nodes, replies, errors)
    
    def has_stopped(self):
        """Checks has federated request has stopped in before finishing the request"""
        return any([s.status == StrategyStatus.STOPPED for s in self.strategies])
    

    def get_stopper_strategies(self):
        """Gets strategies that has stopped federated requests"""
        return [s for s in self.strategies if s.status == StrategyStatus.STOPPED]





class ContinueOnDisconnect(RequestStrategy):
    """Continues collecting results with remaining nodes"""

    def continue_(self) -> bool:
        
        nodes = self.has_not_answered_yet()
        for node in nodes:
            if self.nodes_status[node.id]['status'] == NodeActiveStatus.DISCONNECTED:
                self.nodes.pop(node.id) 
                logger.info("Node has been disconnected during the request. Request will "
                            "continue with remaining nodes")

        return super().continue_()
        
class ContinueOnError(RequestStrategy):
    """Continue collecting results if any node raises an error"""

    status: StrategyStatus = StrategyStatus.UNSET

    def continue_(self) -> bool:

        self.status = StrategyStatus.COMPLETED
        return super().continue_()


class StopOnAnyDisconnect(RequestStrategy):
    """Stops collecting results if a node disconnects"""
    status: StrategyStatus = StrategyStatus.UNSET

    def continue_(self) -> bool:
        
        for node in self.nodes:
            if self.nodes_status[node.id]['status'] == NodeActiveStatus.DISCONNECTED:
                logger.error(f'Node {node.id} has disconnected federated request will be aborted')
                self.status = StrategyStatus.STOPPED
                return False

        self.status = StrategyStatus.COMPLETED
        return super().continue_()


class StopOnAnyError(RequestStrategy):
    """Stops collecting results if a node returns an error"""
    status: StrategyStatus = StrategyStatus.UNSET

    def continue_(self):
        
        for node in self.nodes:
            if self.nodes_status[node.id]['error']:
                logger.error(f'Node {node.id} has returned error federated request will be aborted')
                self.status = StrategyStatus.STOPPED
                return False
        
        self.status = StrategyStatus.COMPLETED
        return super().continue_()

