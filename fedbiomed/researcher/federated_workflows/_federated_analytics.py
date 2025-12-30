# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, Dict, Optional, Union

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.jobs import FARequestJob
from fedbiomed.researcher.requests import Requests


class FederatedAnalytics:
    """
    A class to manage Federated Analytics (FA) workflows within an Experiment.
    FA workflows allow researchers to perform analytics tasks across federated datasets.
    """

    def __init__(
        self,
        fds: FederatedDataset,
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
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        node_ids = self._fds.node_ids()
        if len(node_ids) == 0:
            raise FedbiomedExperimentError(
                "Empty list of nodes for analytics: no nodes replied to original "
                "`federated_analytics_request` or sampling strategy returned an empty list."
            )

        return node_ids

    def mean(
        self, col_names: Optional[list[str | int]] = None
    ) -> Union[Any, Dict[str, Any]]:
        """Compute mean analytics across nodes.

        Returns:
            A dictionary containing the mean analytics results from each node.
        """
        if self._fds is None:
            raise FedbiomedExperimentError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        node_ids = self.get_node_ids()

        # Create FA job
        fa_job = FARequestJob(
            fa_id=self._fa_id,
            fa_args={
                "col_names": col_names if col_names is not None else [],
            },
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Collect training replies and (opt.) optimizer auxiliary variables.
        analytics_replies = fa_job.execute()

        return analytics_replies
