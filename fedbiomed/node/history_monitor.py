'''
send information from node to researcher during the training
'''


from typing import Union, Dict

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
                   metric: Dict[str, Union[int, float]],
                   iteration: int,
                   epoch: int,
                   total_samples: int,
                   batch_samples: int,
                   num_batches: int,
                   test: bool = False,
                   train: bool = False,
                   before_training: Union[bool, None] = None):

        """
        Adds a scalar value to the monitor, and sends an 'AddScalarReply'
        response to researcher

        Args:
            metric (Dict[str, Union[int, float]]):  recorded value
            iteration (int): current epoch iteration.
            epoch (int): current epoch
            total_samples (int):
            batch_samples (int):
            num_batches (int):
            train (bool):
            test (ibool):
            before_training (bool):
            after_training (bool):
        """

        self.messaging.send_message(NodeMessages.reply_create({
            'node_id': environ['NODE_ID'],
            'job_id': self.job_id,
            'researcher_id': self.researcher_id,
            'train': train,
            'test': test,
            'before_training': before_training,
            'metric': metric,
            'iteration': iteration,
            'epoch': epoch,
            'total_samples': total_samples,
            'batch_samples': batch_samples,
            'num_batches': num_batches,
            'command': 'add_scalar'
        }).get_dict(), client='monitoring')