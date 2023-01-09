# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''Send information from node to researcher during the training
'''

from typing import Union, Dict

from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.node.environ import environ


class HistoryMonitor:
    """Send information from node to researcher during the training
    """

    def __init__(self,
                 job_id: str,
                 researcher_id: str,
                 client: Messaging):
        """Simple constructor for the class.

        Args:
            job_id: TODO
            researcher_id: TODO
            client: TODO
        """
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.messaging = client

    def add_scalar(
            self,
            metric: Dict[str, Union[int, float]],
            iteration: int,
            epoch: int,
            total_samples: int,
            batch_samples: int,
            num_batches: int,
            num_samples_trained: int = None,
            train: bool = False,
            test: bool = False,
            test_on_global_updates: bool = False,
            test_on_local_updates: bool = False
    ) -> None:
        """Adds a scalar value to the monitor, and sends an 'AddScalarReply'
            response to researcher.

        Args:
            metric:  recorded value
            iteration: current epoch iteration.
            epoch: current epoch
            total_samples: TODO
            batch_samples: TODO
            num_batches: TODO
            num_samples_trained: TODO
            train: TODO
            test: TODO
            test_on_global_updates: TODO
            test_on_local_updates: TODO

        """

        self.messaging.send_message(NodeMessages.reply_create({
            'node_id': environ['NODE_ID'],
            'job_id': self.job_id,
            'researcher_id': self.researcher_id,
            'train': train,
            'test': test,
            'test_on_global_updates': test_on_global_updates,
            'test_on_local_updates': test_on_local_updates,
            'metric': metric,
            'iteration': iteration,
            'epoch': epoch,
            'num_samples_trained': num_samples_trained,
            'total_samples': total_samples,
            'batch_samples': batch_samples,
            'num_batches': num_batches,
            'command': 'add_scalar'
        }).get_dict(), client='monitoring')
