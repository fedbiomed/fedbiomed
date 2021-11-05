from typing import Union

from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.node.environ import environ


class HistoryMonitor:
    def __init__(self,
                 job_id: str,
                 researcher_id: str,
                 client: Messaging):
        self.history = {}
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.messaging = client

    def add_scalar(self, key: str, value: Union[int, float], iteration: int, epoch: int ):
        """Adds a scalar value to the monitor, and sends an 'AddScalarReply'
        response to researcher

        Args:
            key (str): name value in logger to keep track with
            value (Union[int, float]):  recorded value
            iteration (int): current epoch iteration.
            epoch (int): current epoch
        """

        # Keeps history of the scalar values. Please see Round.py where it is called
        try:
            self.history[key][iteration] = value
        except (KeyError, AttributeError):
            self.history[key] = {iteration: value}

        self.messaging.send_message(NodeMessages.reply_create({
                                                               'node_id': environ['NODE_ID'],
                                                               'job_id': self.job_id,
                                                               'researcher_id': self.researcher_id,
                                                               'key' : key,
                                                               'value': value,
                                                               'iteration': iteration,
                                                               'epoch' : epoch,
                                                               'command': 'add_scalar'
                                                               }).get_dict(),
                                                               client='monitoring'
                                                               )
