"""Federated averaring Aggregator."""

from typing import List

from declearn.model.api import Model, Vector
from fedbiomed.researcher.aggregators.aggregator import Aggregator


class FedAverage(Aggregator):
    """
    Defines the Federated averaging strategy
    """

    aggregator_name = "FedAverage"

    def aggregate(
        self,
        global_model: Model,
        local_model_params: List[Vector],
        weights: List[float],
    ) -> Vector:
        """Aggregate local model parameters and update global model.

        Args:
            global_model: Reference Model handled by the researcher, the
                weights from which are to be updated.
            local_model_params: List of model parameters received from each node.
            weights: List of node-wise weights.

        Returns:
            params: Aggregated parameters, as a declearn NumpyVector.
        """
        # Compute the weighted average of local parameters.
        weights = self.normalize_weights(weights)
        average = sum(p * w for p, w in zip(local_model_params, weights))
        # Compute the average update (rather than weights).
        updates = global_model.get_weights() - average
        # Use the optimizer to refine and apply these updates.
        self.optim.apply_gradients(global_model, updates)
        # Gather and return the updated weights.
        return global_model.get_weights()
