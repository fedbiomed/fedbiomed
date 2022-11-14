"""Monitorer to send information from node to researcher during training."""


from typing import Dict, Union

from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging


class HistoryMonitor:
    """Util to send information from node to researcher during training."""

    def __init__(
            self,
            job_id: str,
            node_id: str,
            researcher_id: str,
            client: Messaging,
        ) -> None:
        """Simple constructor for the class.

        Args:
            job_id: TODO
            node_id: id of the node running the job.
            researcher_id: TODO
            client: TODO
        """
        self.job_id = job_id
        self.node_id = node_id
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
            train: bool = False,
            test: bool = False,
            test_on_global_updates: bool = False,
            test_on_local_updates: bool = False,
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
            train: TODO
            test: TODO
            test_on_global_updates: TODO

        """
        message = NodeMessages.reply_create({
            'node_id': self.node_id,
            'job_id': self.job_id,
            'researcher_id': self.researcher_id,
            'train': train,
            'test': test,
            'test_on_global_updates': test_on_global_updates,
            'test_on_local_updates': test_on_local_updates,
            'metric': metric,
            'iteration': iteration,
            'epoch': epoch,
            'total_samples': total_samples,
            'batch_samples': batch_samples,
            'num_batches': num_batches,
            'command': 'add_scalar'
        })
        self.messaging.send_message(message.get_dict(), client='monitoring')
