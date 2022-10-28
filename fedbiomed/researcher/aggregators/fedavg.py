"""Federated averaring Aggregator."""

from typing import List

from declearn.model.api import Vector

from fedbiomed.researcher.aggregators.aggregator import Aggregator


class FedAverage(Aggregator):
    """
    Defines the Federated averaging strategy
    """

    def __init__(self) -> None:
        """Construct `FedAverage` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        """
        super().__init__()
        self.aggregator_name = "FedAverage"

    def aggregate(
            self,
            model_params: List[Vector],
            weights: List[float],
        ) -> Vector:
        """Aggregate model parameters.

        Args:
            model_params: List of model parameters received from each node.
            weights: List of node-wise weights.

        Returns:
            params: Aggregated parameters, as a declearn Vector.
        """
        weights = self.normalize_weights(weights)
        updates = sum(p * w for p, w in zip(model_params, weights))
        return updates  # type: ignore  # edge case: 0 on empty list
