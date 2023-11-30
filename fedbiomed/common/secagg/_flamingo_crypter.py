# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


import time

from typing import List, Union

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers, VEParameters
from fedbiomed.common.logger import logger


from .utils import quantize, \
    reverse_quantize, \
    apply_weighing, \
    apply_average
from ._flamingo import Flamingo

class FlamingoCrypter:
    """Secure aggregation encryption and decryption manager.

    This class is responsible for encrypting model parameters using Joye-Libert secure
    aggregation scheme. It also aggregates encrypted model parameters and decrypts
    to retrieve final model parameters as vector. This vector can be loaded into model
    by converting it proper format for the framework.
    """

    def __init__(self) -> None:
        """Constructs ParameterEncrypter"""
        self._flamingo = Flamingo()
        self.setup_done = False

    def encrypt(
            self,
            current_round: int,
            my_node_id: int,
            node_ids: List[int],
            params: List[float],
            clipping_range: Union[int, None] = None,
            weight: int = None,
    ) -> List[int]:
        """
        TODO: Add docstring
        """
        start = time.process_time()

        if not isinstance(params, list):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: Expected argument `params` type list but got {type(params)}"
            )

        if not all([isinstance(p, float) for p in params]):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: The parameters to encrypt should list of floats. "
                f"There are one or more than a value that is not type of float."
            )

        # first we quantize the parameters, and we get params in the range [0, 2^VEParameters.TARGET_RANGE]
        params = quantize(weights=params,
                          clipping_range=clipping_range)

        # we multiply the parameters with the weight, and we get params in the range [0, 2^(VEParameters.TARGET_RANGE + VEParameters.MAX_WEIGHT_RANGE)]
        # check if weight if num_bits of weight is less than VEParameters.WEIGHT_RANGE
        if weight is not None:
            if 2**weight.bit_length() > VEParameters.WEIGHT_RANGE:
                raise FedbiomedSecaggCrypterError(
                    f"{ErrorNumbers.FB624.value}: The weight is too large. The weight should be less than "
                    f"{VEParameters.WEIGHT_RANGE}."
                )
        if weight is not None:
            params = apply_weighing(params, weight)

        if not self.setup_done:
            self._flamingo.setup_pairwise_secrets(my_node_id=my_node_id, nodes_ids=node_ids, num_params=len(params))
            self.setup_done = True
        
        try:
            # Encrypt parameters
            encrypted_params: List[int] = self._flamingo.protect(
                current_round=current_round,
                params=params,
                node_ids=node_ids
            )
        except (TypeError, ValueError) as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value} Error during parameter encryption. {exp}") from exp
        
        time_elapsed = time.process_time() - start
        logger.debug(f"Encryption of the parameters took {time_elapsed} seconds.")

        return [int(e_p) for e_p in encrypted_params]

    def aggregate(
            self,
            list_params: List[List[int]],
            num_nodes: int,
            total_sample_size: int,
            clipping_range: Union[int, None] = None
    ) -> List[float]:
        """
        TODO: Add docstring
        """
        start = time.process_time()

        if len(list_params) != num_nodes:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: Num of parameters that are received from nodes "
                f"does not match the number of nodes has been set for the encrypter. There might "
                f"be some nodes did not answered to training request or num of clients of "
                "`ParameterEncrypter` has not been set properly before train request.")

        if not isinstance(list_params, list) or not all([isinstance(p, list) for p in list_params]):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: The parameters to aggregate should "
                                              f"list containing list of parameters")

        if not all([all([isinstance(p_, int) for p_ in p]) for p in list_params]):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: Invalid parameter type. The parameters "
                                              f"should be type of integers.")
        
        try:
            sum_of_params = self._flamingo.aggregate(
                params=list_params
            )
        except (ValueError, TypeError) as e:
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624.value}: The aggregation of encrypted parameters "
                                              f"is not successful: {e}")

        # TODO implement weighted averaging here or in `self._jls.aggregate`
        # Reverse quantize and division (averaging)
        logger.info(f"Aggregating parameters from {len(list_params)} nodes.")
        aggregated_params = apply_average(sum_of_params, total_sample_size)

        aggregated_params: List[float] = reverse_quantize(
            aggregated_params,
            clipping_range=clipping_range
        )
        time_elapsed = time.process_time() - start
        logger.debug(f"Aggregation is completed in {round(time_elapsed, ndigits=2)} seconds.")

        return aggregated_params


    