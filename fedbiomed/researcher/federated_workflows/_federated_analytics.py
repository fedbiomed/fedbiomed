# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from fedbiomed.common.analytics._aggregators import (
    aggregate_mean,
)
from fedbiomed.common.constants import Stats
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import FAReply
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.jobs import FARequestJob
from fedbiomed.researcher.requests import Requests


class FAResult:
    """Class to handle results from Federated Analytics.

    This class wraps the results received from nodes and provides methods to
    access them independently or to aggregate them.
    """

    _modalities: List[str] = None
    _stat_names: List[str] = None

    def __init__(
        self,
        replies: Dict[str, FAReply],
        aggregators: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Constructor.

        Args:
            replies: A dictionary containing the analytics results from each node.
            aggregator: A dictionary of functions to apply on the results for aggregation.
        """
        # Validate that we have replies
        if not replies:
            raise FedbiomedError("No replies provided to FAResult.")

        # Validate reply structure and consistency
        for node_id, reply in replies.items():
            self._validate_reply(node_id, reply)

        self._results = {node_id: reply.output for node_id, reply in replies.items()}

        # Validate aggregators
        self._validate_aggregators(aggregators)
        self._aggregators = aggregators

    def _validate_aggregators(self, aggregators: Optional[Dict[str, Callable]]) -> None:
        """Validates the aggregators provided."""
        if aggregators is None:
            return
        if not isinstance(aggregators, dict):
            raise FedbiomedError(
                "Aggregators should be provided as a dictionary of callables."
            )
        if not aggregators:
            raise FedbiomedError("Aggregators dictionary cannot be empty.")
        for stat_name, func in aggregators.items():
            if not callable(func):
                raise FedbiomedError(f"Aggregator for '{stat_name}' is not callable.")

    def _validate_reply(self, node_id: str, reply: FAReply) -> None:
        """Validates the structure and consistency of a reply.

        Args:
            node_id: The ID of the node that sent the reply.
            reply: The reply object to validate.
        """
        # Check structure
        if not isinstance(reply.output, dict):
            raise FedbiomedError(f"Node {node_id} returned invalid output format.")
        if not reply.output:
            raise FedbiomedError(f"Output for node {node_id} is empty.")

        # Initialize modalities on first reply
        if self._modalities is None:
            self._modalities = sorted(list(reply.output.keys()))

        # Check modalities consistency
        if sorted(list(reply.output.keys())) != self._modalities:
            raise FedbiomedError("Nodes present inconsistent modalities.")

        # Validate each modality's stats
        for modality, stats in reply.output.items():
            if not isinstance(stats, dict):
                raise FedbiomedError(
                    f"Node {node_id}, modality {modality} output is not a dictionary."
                )
            if not stats:
                raise FedbiomedError(
                    f"Node {node_id}, modality {modality} has empty statistics."
                )

            # Initialize stat names on first modality
            if self._stat_names is None:
                self._stat_names = sorted(list(stats.keys()))

            # Check stat names consistency
            if sorted(list(stats.keys())) != self._stat_names:
                raise FedbiomedError(
                    f"Node {node_id}, modality {modality} has inconsistent statistics."
                )

    @property
    def modalities(self) -> List[str]:
        return self._modalities

    @property
    def stat_names(self) -> List[str]:
        return self._stat_names

    @property
    def results(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Returns nested dict [node_id][modality][stat_name][value]"""
        return self._results

    @property
    def values(self) -> Dict[str, Dict[str, List[Any]]]:
        """Returns values grouped by modality and statistic.

        Returns:
            A nested dictionary where keys are modalities and statistics, and values are lists of
            collected values from all nodes.
            Format: {modality: {stat_name: [value_node_1, value_node_2, ...]}}
        """
        values = {
            mod: {stat: [] for stat in self._stat_names} for mod in self._modalities
        }

        for result in self._results.values():
            for mod in self._modalities:
                for stat in self._stat_names:
                    values[mod][stat].append(result[mod][stat])
        return values

    def aggregate(self) -> Dict[str, Any]:
        """Aggregates the results from all nodes.

        Returns:
             The aggregated results.
        """
        if self._aggregators is None:
            logger.info("No parameter 'aggregators' available, returning raw values.")
            return self.values

        # Initialize output structure
        output = {modality: {} for modality in self._modalities}

        # Remove nodes that returned errors or empty results
        for modality, stats in self.values.items():
            for stat_name, aggregator in self._aggregators.items():
                try:
                    output[modality][stat_name] = aggregator(**stats)
                except Exception as e:
                    logger.warning(
                        f"Aggregation failed for modality '{modality}', statistic '{stat_name}': {e}"
                    )

        return output


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

    def _compute_analytics(
        self,
        stats: Union[str, List[str]],
        dataset_args: Optional[dict] = None,
        dataset_schema: Optional[str | list[str | dict]] = None,
        fa_args: Optional[dict] = None,
    ) -> Union[Any, Dict[str, Any]]:
        """Compute analytics across nodes.

        Args:
            stats: The type of analytics to compute (string or list of strings).
            dataset_args: Dataset arguments.
            fa_args: Federated analytics arguments.

        Returns:
            A dictionary containing the analytics results from each node.
        """
        if self._fds is None:
            raise FedbiomedError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        # TODO: self._validate_dataset_arguments(dataset_args)
        node_ids = self.get_node_ids()

        # Create FA job
        fa_job = FARequestJob(
            fa_id=self._fa_id,
            fa_args=fa_args,
            stats=stats,
            dataset_args=dataset_args,
            dataset_schema=dataset_schema,
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Collect replies
        analytics_replies, errors = fa_job.execute()

        # TODO: define error handling strategy
        for node_id, error in errors.items():
            logger.warning(
                "Error message received during analytics request for node "
                f"{node_id} - {error.errnum}: {error.extra_msg}"
            )
            # _ = self._fds._data.pop(node_id)

        return analytics_replies, errors

    def mean(
        self,
        dataset_args: Optional[dict] = None,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> Union[Any, Dict[str, Any]]:
        """Returns FAResult object containing mean for each node."""
        # Collect replies
        replies, _ = self._compute_analytics(
            stats=Stats.MEAN.value,
            dataset_args=dataset_args,
            dataset_schema=dataset_schema,
        )
        # Define aggregators
        aggregators = {
            "mean": aggregate_mean,
        }
        return FAResult(replies, aggregators)
