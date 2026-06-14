# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Send information from node to researcher during the training
"""

from typing import Callable, Dict, Union

from fedbiomed.common.logger import logger
from fedbiomed.common.message import FeedbackMessage, Scalar


class HistoryMonitor:
    """Send information from node to researcher during the training"""

    def __init__(
        self,
        node_id: str,
        node_name: str,
        experiment_id: str,
        researcher_id: str,
        send: Callable,
    ):
        """Simple constructor for the class.

        Args:
            node_id: Unique ID of this node
            node_name: Human-readable name of this node
            experiment_id: Unique identifier of the running experiment
            researcher_id: Unique identifier of the researcher receiving feedback
            send: Callable used to transmit a FeedbackMessage
        """
        self._node_id = node_id
        self._node_name = node_name
        self.experiment_id = experiment_id
        self.researcher_id = researcher_id
        self.send = send

        logger.debug(
            "Initialized history monitor for: node=%s experiment=%s researcher=%s",
            self._node_id,
            self.experiment_id,
            self.researcher_id,
        )

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
        test_on_local_updates: bool = False,
    ) -> None:
        """Adds a scalar value to the monitor, and sends an 'AddScalarReply'
            response to researcher.

        Args:
            metric: Recorded metric values (for example loss or accuracy)
            iteration: Current iteration index within the training process
            epoch: Current epoch index
            total_samples: Total number of samples in the dataset
            batch_samples: Number of samples in the current batch
            num_batches: Number of batches for one epoch
            num_samples_trained: Cumulative number of samples used for training
            train: Whether this metric is emitted during training
            test: Whether this metric is emitted during validation/testing
            test_on_global_updates: Whether test metrics use globally updated params
            test_on_local_updates: Whether test metrics use locally updated params

        Returns:
            None
        """
        self.send(
            FeedbackMessage(
                researcher_id=self.researcher_id,
                scalar=Scalar(
                    **{
                        "node_id": self._node_id,
                        "node_name": self._node_name,
                        "experiment_id": self.experiment_id,
                        "train": train,
                        "test": test,
                        "test_on_global_updates": test_on_global_updates,
                        "test_on_local_updates": test_on_local_updates,
                        "metric": metric,
                        "iteration": iteration,
                        "epoch": epoch,
                        "num_samples_trained": num_samples_trained,
                        "total_samples": total_samples,
                        "batch_samples": batch_samples,
                        "num_batches": num_batches,
                    }
                ),
            )
        )
