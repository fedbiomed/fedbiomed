"""
"""

from typing import Dict, Union, Mapping

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_standardization


class FedStandard(Aggregator):
    """
    Defines the Federated averaging strategy
    """

    def __init__(self):
        """Construct `FedStandard` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        """
        super(FedStandard, self).__init__()
        self.aggregator_name = "FedStandard"

    def aggregate(
            self,
            model_params: Dict[str, Dict[str, Union['torch.Tensor', 'numpy.ndarray']]],
            weights: Dict[str, float],
            *args,
            **kwargs
    ) -> Mapping[str, Union['torch.Tensor', 'np.ndarray']]:
            #self, model_params: list, weights: list, *args, **kwargs) -> Dict:
        
        """ Aggregates  local models sent by participating nodes into a global model, following Federated Averaging
        strategy.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """

        model_params_processed = list()
        for node_id, params in model_params.items():

            if node_id not in weights:
                raise FedbiomedAggregatorError(
                    f"{ErrorNumbers.FB401.value}. Can not find corresponding calculated weight for the "
                    f"node {node_id}. Aggregation is aborted."
                )

            model_params_processed.append(params)
        #model_params_processed = [list(model_param.values())[0] for model_param in model_params]
        return federated_standardization(model_params_processed, weights)
