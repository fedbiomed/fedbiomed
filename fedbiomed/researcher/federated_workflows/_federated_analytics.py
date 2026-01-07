# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, Dict, List, Union

import numpy as np

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

    def _compute_analytics(
        self, analytics_type: str, dataset_args: dict = None, fa_args: dict = None
    ) -> Union[Any, Dict[str, Any]]:
        """Compute analytics across nodes.

        Args:
            analytics_type: The type of analytics to compute.
            dataset_args: Dataset arguments.
            fa_args: Federated analytics arguments.

        Returns:
            A dictionary containing the analytics results from each node.
        """
        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        self._validate_if_dataset_has_analytics(analytics_type)
        self._validate_dataset_arguments(dataset_args)
        node_ids = self.get_node_ids()

        # Create FA job
        fa_job = FARequestJob(
            fa_id=self._fa_id,
            fa_args=fa_args,
            analytics_type=analytics_type,
            dataset_args=dataset_args,
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Collect replies
        analytics_replies = fa_job.execute()

        return analytics_replies

<<<<<<< HEAD
    def basic_stats(
        self, dataset_args: dict = None, fa_args: dict = None
    ) -> Union[Any, Dict[str, Any]]:
        """Returns a dict containing the basic analytics for each node (min, max, count, mean, std)."""
        return self._compute_analytics(
            AnalyticsTypes.BASIC_STATS.value, dataset_args, fa_args
        )

    def min_max(
        self, dataset_args: dict = None, fa_args: dict = None
    ) -> Union[Any, Dict[str, Any]]:
        """Returns a dict containing min and max for each node."""
        return self._compute_analytics(
            AnalyticsTypes.MIN_MAX.value, dataset_args, fa_args
        )

    def mean(
        self, dataset_args: dict = None, fa_args: dict = None
    ) -> Union[Any, Dict[str, Any]]:
        """Returns a dict containing min and max for each node."""
        return self._compute_analytics(AnalyticsTypes.MEAN.value, dataset_args, fa_args)
=======
    def _get_global_minmax(
        self, dataset_args: dict = None
    ) -> Dict[str, tuple[float, float]]:
        """Get global min/max values across all nodes for each column.

        This method sends a request to all nodes to compute local min/max values,
        then aggregates them to determine global boundaries for bin creation.

        Args:
            dataset_args: Dataset arguments for minmax computation

        Returns:
            A dictionary mapping column names to (min, max) tuples
        """
        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        self._validate_dataset_arguments(dataset_args)
        node_ids = self.get_node_ids()

        # Create FA job to get min/max from each node
        fa_job = FARequestJob(
            fa_id=self._fa_id,
            fa_args={},
            analytics_type=AnalyticsTypes.MINMAX.value,
            dataset_args=dataset_args,
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Collect minmax replies from all nodes
        minmax_replies, errors = fa_job.execute()

        if errors:
            raise FedbiomedError(
                f"Errors occurred while computing global min/max: {errors}"
            )

        # Aggregate min/max across all nodes
        global_minmax = {}
        for _, reply in minmax_replies.items():
            node_minmax = reply.output
            for col, (node_min, node_max) in node_minmax.items():
                if col not in global_minmax:
                    global_minmax[col] = (node_min, node_max)
                else:
                    global_min, global_max = global_minmax[col]
                    global_minmax[col] = (
                        min(global_min, node_min),
                        max(global_max, node_max),
                    )

        return global_minmax

    def _create_bins(
        self,
        global_minmax: Dict[str, tuple[float, float]],
        num_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Create uniform bins for each column based on global min/max.

        Args:
            global_minmax: Dictionary mapping column names to (min, max) tuples
            num_bins: Number of bins to create for each column

        Returns:
            Dictionary mapping column names to bin_edges arrays
        """
        bin_edges = {}
        for col, (min_val, max_val) in global_minmax.items():
            # Add small margin to include max value in last bin
            margin = (max_val - min_val) * 0.001 if max_val != min_val else 1
            bin_edges[col] = np.linspace(min_val, max_val + margin, num_bins + 1)
        return bin_edges

    def histogram(
        self,
        bin_edges: Union[List, Dict[str, List]] = None,
        num_bins: int = 10,
        dataset_args: dict = None,
        fa_args: dict = None,
    ) -> Union[Any, Dict[str, Any]]:
        """Compute histogram analytics across nodes.

        Args:
            bin_edges: Pre-computed bin edges. Can be:
                - 1D numpy array: applied to all columns (for images)
                - Dict mapping column names to bin_edges: column-specific bins
                - None: will compute from global min/max (for tabular data)
            num_bins: Number of bins to create if bin_edges is None. Default is 10.
            dataset_args: Dataset arguments for histogram computation

        Returns:
            A dictionary containing the histogram analytics results from each node.
        """
        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        print("Validating analytics")
        self._validate_if_dataset_has_analytics(AnalyticsTypes.HISTOGRAM.value)
        print("Validating dataset arguments")
        self._validate_dataset_arguments(dataset_args)
        node_ids = self.get_node_ids()

        # If bin_edges not provided, compute from global min/max
        if bin_edges is None:
            print("Bin edges is None, computing global minmax")
            global_minmax = self._get_global_minmax(dataset_args)
            print("Global minmax computed:", global_minmax)
            print("Setting bin edges")
            bin_edges = self._create_bins(global_minmax, num_bins)

        # Prepare fa_args with bin_edges
        fa_args = fa_args or {}
        fa_args["bin_edges"] = bin_edges
        print("FA args prepared:", fa_args)

        # Create FA job
        fa_job = FARequestJob(
            fa_id=self._fa_id,
            fa_args=fa_args,
            analytics_type=AnalyticsTypes.HISTOGRAM.value,
            dataset_args=dataset_args,
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Collect histogram replies
        analytics_replies, errors = fa_job.execute()

        return analytics_replies, errors
>>>>>>> dd3aa21e (First working draft for histogram for Tabular Dataset)
