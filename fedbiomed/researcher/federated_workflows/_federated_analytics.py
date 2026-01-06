# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, Dict, Union

from fedbiomed.common.analytics import (
    validate_dataset_arguments_for_fa,
)
from fedbiomed.common.constants import AnalyticsTypes, DatasetTypes
from fedbiomed.common.dataset import DATASET_CLASSES_PER_TYPE
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
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

    def _validate_if_dataset_has_analytics(self, analytics_type: str) -> DatasetTypes:
        """Get the dataset type from the federated dataset.

        Returns:
            The dataset type
        """

        # Extract dataset types from the first self._fds.data() entry
        type_ = next(iter(self._fds.data().values())).get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(type_)

        # Check if dataset class has requested analytics implemented
        dataset_cls = DATASET_CLASSES_PER_TYPE[dataset_type]
        if not hasattr(dataset_cls, analytics_type):
            raise FedbiomedError(
                f"Dataset type '{dataset_type.value}' does not support "
                f"analytics type '{analytics_type}'."
            )

        return dataset_type

    def _validate_dataset_arguments(self, dataset_args: dict) -> None:
        """Validate dataset arguments for federated analytics.

        Args:
            dataset_args: Dataset arguments to validate
        """

        # Extract dataset types from the first self._fds.data() entry
        type_ = next(iter(self._fds.data().values())).get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(type_)

        validate_dataset_arguments_for_fa(dataset_args, dataset_type)

    def mean(
        self, dataset_args: dict = None, fa_args: dict = None
    ) -> Union[Any, Dict[str, Any]]:
        """Compute mean analytics across nodes.

        Returns:
            A dictionary containing the mean analytics results from each node.
        """
        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )
            # Extract dataset types from the first self._fds.data() entry

        self._validate_if_dataset_has_analytics(AnalyticsTypes.MEAN.value)
        self._validate_dataset_arguments(dataset_args)
        node_ids = self.get_node_ids()

        # Create FA job
        fa_job = FARequestJob(
            fa_id=self._fa_id,
            fa_args=fa_args,
            analytics_type=AnalyticsTypes.MEAN.value,
            dataset_args=dataset_args,
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Collect training replies and (opt.) optimizer auxiliary variables.
        analytics_replies = fa_job.execute()

        return analytics_replies
