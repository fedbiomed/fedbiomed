"""Code of the researcher. Implements the experiment orchestration"""

from typing import List, Type, TypeVar, Union

import numpy as np
import torch
from gmpy2 import mpz

from fedbiomed.common.constants import VEParameters

from fedbiomed.common.joye_libert_scheme import JLS, ServerKey
from fedbiomed.common.joye_libert_utils import reverse_quantize
from fedbiomed.common.logger import logger

from fedbiomed.common.vector_encoding import VES
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.experiment import Experiment, Type_TrainingPlan

from fedbiomed.researcher.strategies.strategy import Strategy


class ExperimentJL(Experiment):
    def __init__(
        self,
        tags: Union[List[str], str, None] = None,
        nodes: Union[List[str], None] = None,
        training_data: Union[FederatedDataSet, dict, None] = None,
        aggregator: Union[Aggregator, Type[Aggregator], None] = None,
        node_selection_strategy: Union[Strategy, Type[Strategy], None] = None,
        round_limit: Union[int, None] = None,
        model_class: Union[Type_TrainingPlan, str, None] = None,
        model_path: Union[str, None] = None,
        model_args: dict = {},
        training_args: Union[TypeVar("TrainingArgs"), dict, None] = None,
        save_breakpoints: bool = False,
        tensorboard: bool = False,
        experimentation_folder: Union[str, None] = None,
    ):
        super().__init__(
            tags,
            nodes,
            training_data,
            aggregator,
            node_selection_strategy,
            round_limit,
            model_class,
            model_path,
            model_args,
            training_args,
            save_breakpoints,
            tensorboard,
            experimentation_folder,
        )

        self.vector_encoder = VES(
            ptsize=VEParameters.KEY_SIZE.value // 2,
            addops=VEParameters.NUM_CLIENTS.value,
            valuesize=VEParameters.VALUE_SIZE.value,
            vectorsize=VEParameters.VECTOR_SIZE.value,
        )


    def _aggregate_params(self):
        models_ctxt_qt_enc, weights = self._node_selection_strategy.refine(
            self._job.training_replies[self._round_current], self._round_current
        )
        # get the public params, exploiting the EncryptedNumber objects
        public_params = models_ctxt_qt_enc[0][0].pp
        # following section 6.3 [1], s_0 = 0
        jl = JLS(VEParameters.NUM_CLIENTS.value, VE=self.vector_encoder)

        s_0 = mpz(0)
        server_key = ServerKey(param=public_params, key=s_0)
        # configure vector_encoder (using the same conf of the nodes)

        # call aggregation, following section 3.6 eq. 2 [1]
        logger.info("Aggregating models")
        aggregated_params = jl.Agg(
            pp=public_params,
            sk_0=server_key,
            tau=self._round_current,
            list_y_u_tau=models_ctxt_qt_enc,
        )
        logger.info(
            f"Joye-Libert aggregation done, length of aggregated_params: {len(aggregated_params)}"
        )
        # reserve quantization of the aggregated_params, from int to float
        print("aggregated_params", aggregated_params[:10])
        aggregated_params = reverse_quantize(aggregated_params)
        weights = Aggregator.normalize_weights(weights)
        print("weights", weights)
        print("aggregated_params", aggregated_params[:10])
        aggregated_params = aggregated_params * 0.5 #np.average(aggregated_params, weights=0.5, axis=0)

        aggregated_params = torch.as_tensor(aggregated_params).type(torch.DoubleTensor)
        # create dummy_parameters, to recreate a torch.Parameters object which will be filled with aggregated_params
        # which is a list
        dummy_parameters = self.model_instance().parameters()
        # convert a list into a tensor and assign it the right type
        # fill torch.Parameters object with a torch.tensor
        torch.nn.utils.vector_to_parameters(aggregated_params, dummy_parameters)
        # create a dummy model which will be filled using the new dummy_parameters (filled using aggregated_params)
        dummy_model = self.model_instance()
        with torch.no_grad():
            for p_dest, p_src in zip(dummy_model.parameters(), dummy_parameters):
                p_dest.copy_(p_src.data.clone())

        aggregated_params_path = self._job.update_parameters(dummy_model.state_dict())

        return aggregated_params, aggregated_params_path
