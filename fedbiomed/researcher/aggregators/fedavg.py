from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


class FedAverage(Aggregator):
    """ Defines the Federated averaging strategy """

    def __init__(self):
        super(FedAverage, self).__init__()

    def aggregate(self, model_params: list, weights: list):
        weights = self.normalize_weights(weights)
        return federated_averaging(model_params, weights)
