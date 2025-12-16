from typing import Any, Dict, Optional

import numpy as np

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows.jobs._job import Job


class FAResearcherJob(Job):
    """
    TrainingJob is a task for training an ML model on the nodes by executing a
    [TrainingPlan][fedbiomed.common.training_plans.BaseTrainingPlan].
    """

    def __init__(
        self,
        fa_id: str,
        experiment_id: str,
        data: FederatedDataSet,
        node_ids: Dict[str, str],
        fa_args: Optional[dict],
        **kwargs,
    ):
        """Constructor of the class

        Args:
            experiment_id: unique ID of this experiment
            data: metadata of the federated data set
            nodes_state_ids: unique IDs of the node states saved remotely
            fa_args: the request arguments specific to FA jobs
            **kwargs: Named arguments of parent class. Please see
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]
        """
        super().__init__(**kwargs)
        # to be used for `execute()`
        self._fa_id = fa_id
        self._experiment_id = experiment_id
        self._data = data
        self._node_ids = node_ids
        self._fa_args = fa_args

    def execute(self) -> Dict[str, Any]:
        """Execute the FA training job

        Returns:
            A dictionary containing the results of the training job from each node.
        """
        # Build request payload
        payload = {
            "fa_id": self._fa_id,
            "experiment_id": self._experiment_id,
            "data": self._data,
            "node_ids": self._node_ids,
        }
        if self._fa_args:
            payload["fa_args"] = self._fa_args

        # Send request to nodes
        # responses = self._reqs.send_fa_training_request(
        #     nodes=self._nodes,
        #     payload=payload,
        #     policies=self._policies,
        #     keep_files_dir=self._keep_files_dir,
        # )

        # For now, we simulate FA execution with random data
        analytics = np.random.rand(3, 3)

        return payload, analytics
