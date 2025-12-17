import uuid
from typing import Any, Dict, Optional, Union

import numpy as np

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows.jobs import Job
from fedbiomed.researcher.requests import Requests


class FAResearcherJob(Job):
    """
    A class representing a Federated Analytics (FA) job to be executed on federated nodes.
    This class extends the base Job class and includes specific attributes and methods
    for handling FA tasks.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """Dummy class to test FA workflow integration"""
        super().__init__(
            researcher_id=kwargs.get("researcher_id"),
            requests=kwargs.get("requests"),
            nodes=kwargs.get("nodes"),
            keep_files_dir=kwargs.get("keep_files_dir"),
        )

        # to be used for `execute()`
        self._kwargs = kwargs

    def execute(self) -> Dict[str, Any]:
        """Execute the FA training job

        Returns:
            A dictionary containing the results of the training job from each node.
        """

        # For now, we simulate FA execution with random data
        analytics = np.random.rand(3, 3)

        return self._kwargs, analytics


class FederatedAnalytics:
    """
    A class to manage Federated Analytics (FA) workflows within an Experiment.
    FA workflows allow researchers to perform analytics tasks across federated datasets.
    """

    def __init__(
        self,
        fds: FederatedDataSet,
        experiment_id: str,
        researcher_id: str,
        reqs: Requests,
        experimentation_folder: str,
        **kwargs,
    ) -> None:
        """Constructor of the class.

        Args:
            **kwargs: Additional named arguments
        """
        self._fa_id: str = "FA_" + str(uuid.uuid4())  # creating a unique experiment id
        self._fds = fds
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._reqs = reqs
        self._experimentation_folder = experimentation_folder
        self._kwargs = kwargs

    @property
    def fa_id(self) -> str:
        """Get the unique ID of this federated analytics.

        Returns:
            The unique ID of this federated analytics
        """
        return self._fa_id

    def get_node_ids(self) -> list[str]:
        """Get the list of node IDs participating in this federated analytics.

        Returns:
            A list of node IDs
        """
        if self._fds is None:
            raise FedbiomedExperimentError(
                "No defined FederatedDataSet found for FederatedAnalytics."
            )

        node_ids = self._fds.node_ids()
        if len(node_ids) == 0:
            raise FedbiomedExperimentError(
                "Empty list of nodes for analytics: no nodes replied to original "
                "`federated_analytics_request` or sampling strategy returned an empty list."
            )

        return node_ids

    def mean(self, col_names: Optional[list[str | int]]) -> Union[Any, Dict[str, Any]]:
        """Compute mean analytics across nodes.

        Returns:
            A dictionary containing the mean analytics results from each node.
        """
        if self._fds is None:
            raise FedbiomedExperimentError(
                "No defined FederatedDataSet found for FederatedAnalytics."
            )

        node_ids = self.get_node_ids()

        # Create FA job
        fa_job = FAResearcherJob(
            fa_id=self._fa_id,
            fa_args={
                "analytics_method": "mean",
                "col_names": col_names if col_names is not None else [],
            },
            fds=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
            keep_files_dir=self._experimentation_folder,
        )

        print("Nodes replied for analytics: " + str(fa_job.nodes))

        # Collect training replies and (opt.) optimizer auxiliary variables.
        analytics_replies = fa_job.execute()

        return analytics_replies
