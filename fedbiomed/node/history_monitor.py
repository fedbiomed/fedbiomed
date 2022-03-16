'''
send information from node to researcher during the training
'''


from typing import Union

from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.node.environ import environ


class HistoryMonitor:
    def __init__(self,
                 job_id: str,
                 researcher_id: str,
                 client: Messaging):
        """
        simple constructor
        """
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.messaging = client

    def add_scalar(self,
                   key: str,
                   value: Union[int, float],
                   iteration: int,
                   epoch: int,
                   total_samples: int,
                   batch_samples: int,
                   result_for: str,
                   num_batches: int):

        """
        Adds a scalar value to the monitor, and sends an 'AddScalarReply'
        response to researcher

        Args:
            key (str): name value in logger to keep track with
            value (Union[int, float]):  recorded value
            iteration (int): current epoch iteration.
            epoch (int): current epoch
            total_samples (int):
            batch_samples (int):
            result_for (str):
            num_batches (int):
        """

        self.messaging.send_message(NodeMessages.reply_create({
            'node_id': environ['NODE_ID'],
            'job_id': self.job_id,
            'researcher_id': self.researcher_id,
            'result_for': result_for,
            'key': key,
            'value': value,
            'iteration': iteration,
            'epoch': epoch,
            'total_samples': total_samples,
            'batch_samples': batch_samples,
            'num_batches': num_batches,
            'command': 'add_scalar'
        }).get_dict(), client='monitoring')

        researcher_id: str
        node_id: str
        job_id: str
        key: str
        value: float
        epoch: int
        total_samples: int
        batch_samples: int
        num_batches: int
        result_for: str
        iteration: int
        command: str