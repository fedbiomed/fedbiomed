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
from fedbiomed.common.logger import logger
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
        logger.debug("Validating analytics")

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
        logger.debug("Validating dataset arguments")

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

    def _get_global_minmax(
        self, dataset_args: dict = None
    ) -> Dict[str, tuple[float, float]]:
        """Get global min/max values across all nodes for each column.

        This method sends a request to all nodes to compute local min/max values,
        then aggregates them to determine global boundaries for bin creation.

        Args:
            dataset_args: Dataset arguments for minmax computation

        Returns:
            Either a dictionary mapping column names to (min, max) tuples such as:
                {"column_1": (min_value, max_value), "column_2": (min_value, max_value), ...}
            or a dict with one key "result (such as 'pixel_values')" mapping to (min, max) tuple for non-tabular data.

        Raises:
            FedbiomedError: if errors occur during minmax computation on nodes
        """
        logger.debug("Bin edges is None, computing global minmax")

        # Collect minmax replies from all nodes
        minmax_replies, errors = self.min_max(dataset_args)

        if errors:
            raise FedbiomedError(
                f"Errors occurred while computing global min/max: {errors}"
            )

        # Aggregate min/max across all nodes
        global_minmax = {}
        for _node_id, reply in minmax_replies.items():
            node_minmax = reply.output
            for _col_or_result_key, min_max_dict in node_minmax.items():
                if _col_or_result_key not in global_minmax:
                    global_minmax[_col_or_result_key] = (
                        min_max_dict["min"],
                        min_max_dict["max"],
                    )
                else:
                    global_min, global_max = global_minmax[_col_or_result_key]
                    global_minmax[_col_or_result_key] = (
                        min(global_min, min_max_dict["min"]),
                        max(global_max, min_max_dict["max"]),
                    )

        logger.debug("Global minmax computed:", global_minmax)

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
        logger.debug("Setting bin edges")

        bin_edges = {}
        for col_or_result_key, (min_val, max_val) in global_minmax.items():
            # Add small margin to include max value in last bin
            margin = (max_val - min_val) * 0.001 if max_val != min_val else 1
            bin_edges[col_or_result_key] = np.linspace(
                min_val, max_val + margin, num_bins + 1
            )
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
                - 1D numpy array: applied to all columns (for both tabular data or all pixels in images)
                - Dict mapping column names to bin_edges: column-specific bins
                - None: will compute from global min/max (for tabular data)
            num_bins: Number of bins to create if bin_edges is None. Default is 10.
            dataset_args: Dataset arguments for histogram computation

        Returns:
            Tuple containing:
            - aggregated_histogram: Dictionary with aggregated counts from all nodes
            - node_histograms: Dictionary with per-node results
            - errors: Any errors from node execution
        """
        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        self._validate_if_dataset_has_analytics(AnalyticsTypes.HISTOGRAM.value)
        self._validate_dataset_arguments(dataset_args)

        # If bin_edges not provided, compute from global min/max
        if bin_edges is None:
            global_minmax = self._get_global_minmax(dataset_args)
            bin_edges = self._create_bins(global_minmax, num_bins)

        # Prepare fa_args with bin_edges
        fa_args = fa_args or {}
        fa_args["bin_edges"] = bin_edges
        logger.debug("FA args prepared:", fa_args)

        # Collect histogram replies
        node_histograms, errors = self._compute_analytics(
            AnalyticsTypes.HISTOGRAM.value, dataset_args, fa_args
        )

        # Aggregate histograms from all nodes
        # TODO: Add common aggregation strategies (e.g., weighted by sample size)
        # and delegate below part to specific strategy
        aggregated_histogram = {}
        for _node_id, node_result in node_histograms.items():
            hist_data = node_result.output
            for col_name, counts in hist_data.items():
                if col_name not in aggregated_histogram:
                    aggregated_histogram[col_name] = np.array(counts)
                else:
                    aggregated_histogram[col_name] += np.array(counts)

        return aggregated_histogram, node_histograms, errors

    def quantile(
        self,
        q: Union[float, List[float]] = 0.5,
        bin_edges: Union[List, Dict[str, List]] = None,
        num_bins: int = 10,
        dataset_args: dict = None,
        fa_args: dict = None,
    ) -> tuple:
        """Compute quantile values across federated nodes.

        Args:
            q: Quantile value(s) to compute. Can be:
                - float between 0 and 1 (e.g., 0.5 for median)
                - List of floats for multiple quantiles
            bin_edges: Pre-computed bin edges. Can be:
                - 1D numpy array: applied to all columns (for images)
                - Dict mapping column names to bin_edges: column-specific bins
                - None: will compute from global min/max (for tabular data)
            num_bins: Number of bins to create if bin_edges is None. Default is 10.
            dataset_args: Dataset arguments for quantile computation
            fa_args: Additional federated analytics arguments

        Returns:
            Tuple containing:
            - aggregated_quantiles: Dictionary with aggregated quantile values from all nodes
            - node_quantiles: Dictionary with per-node results
            - errors: Any errors from node execution

        Raises:
            FedbiomedError: if no FederatedDataset is defined
        """

        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        self._validate_if_dataset_has_analytics(AnalyticsTypes.QUANTILE.value)
        self._validate_dataset_arguments(dataset_args)

        # If bin_edges not provided, compute from global min/max
        if bin_edges is None:
            global_minmax = self._get_global_minmax(dataset_args)
            bin_edges = self._create_bins(global_minmax, num_bins)

        # Prepare fa_args with bin_edges and quantile values
        fa_args = fa_args or {}
        fa_args["bin_edges"] = bin_edges
        fa_args["q"] = q
        logger.debug("FA args prepared:", fa_args)

        # Collect quantile replies
        node_quantiles, errors = self._compute_analytics(
            AnalyticsTypes.QUANTILE.value, dataset_args, fa_args
        )

        # Aggregate quantiles from all nodes
        # TODO: Change aggregation strategy (maybe weighted by sample size)
        # TODO: Delegate below part to a common class for aggregation strategies
        aggregated_quantiles = {}
        for _node_id, node_result in node_quantiles.items():
            quant_data = node_result.output
            for col_name, values in quant_data.items():
                if col_name not in aggregated_quantiles:
                    # Initialize with lists to collect values from all nodes
                    aggregated_quantiles[col_name] = {
                        q_key: [q_val] for q_key, q_val in values.items()
                    }
                else:
                    # Add values from this node to the lists
                    for q_key, q_val in values.items():
                        if q_key not in aggregated_quantiles[col_name]:
                            aggregated_quantiles[col_name][q_key] = []
                        aggregated_quantiles[col_name][q_key].append(q_val)

        # Average quantile values across nodes
        for col_name in aggregated_quantiles:
            for q_key in aggregated_quantiles[col_name]:
                values_list = aggregated_quantiles[col_name][q_key]
                aggregated_quantiles[col_name][q_key] = float(np.mean(values_list))

        return aggregated_quantiles, node_quantiles, errors
