# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


import time

from typing import List, Union
from gmpy2 import mpz

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers, VEParameters
from fedbiomed.common.logger import logger

from ._jls import JoyeLibert, \
    EncryptedNumber, \
    ServerKey, \
    UserKey, \
    FDH, \
    PublicParam, \
    quantize, \
    reverse_quantize, \
    multiply, \
    divide,\
    reverse_quantize, \
    multiply, \
    divide


class SecaggCrypter:
    """Secure aggregation encryption and decryption manager.

    This class is responsible for encrypting model parameters using Joye-Libert secure
    aggregation scheme. It also aggregates encrypted model parameters and decrypts
    to retrieve final model parameters as vector. This vector can be loaded into model
    by converting it proper format for the framework.
    """

    def __init__(self) -> None:
        """Constructs ParameterEncrypter"""
        self._jls = JoyeLibert()

    @staticmethod
    def _setup_public_param(biprime: int) -> PublicParam:
        """Creates public parameter for encryption

        Returns:
            Public parameters
        """

        key_size = VEParameters.KEY_SIZE
        biprime = mpz(biprime)

        fdh = FDH(bits_size=key_size,
                  n_modulus=biprime * biprime)

        return PublicParam(n_modulus=biprime,
                           bits=key_size // 2,
                           hashing_function=fdh.H)

    def encrypt(
            self,
            num_nodes: int,
            current_round: int,
            params: List[float],
            key: int,
            biprime: int,
            clipping_range: Union[int, None] = None,
            weight: int = None,
    ) -> List[int]:
        """Encrypts model parameters.

        Args:
            num_nodes: Number of nodes that is expected to encrypt parameters for aggregation
            current_round: Current round of federated training
            params: List of flatten parameters
            key: Key to encrypt
            biprime: Prime number to create public parameter
            weight: Weight for the params
            clipping_range: Clipping-range for quantization of float model parameters. Clipping range
                must grater than minimum model parameters

        Returns:
            List of encrypted parameters

        Raises:
            FedbiomedSecaggCrypterError: bad parameters
            FedbiomedSecaggCrypterError: encryption issue
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

        # Make use the key is instance of
        if not isinstance(key, int):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: The argument `key` must be integer"
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
        params = self._apply_weighing(params, weight)


        public_param = self._setup_public_param(biprime=biprime)

        # Instantiates UserKey object
        key = UserKey(public_param, key)

        try:
            # Encrypt parameters
            encrypted_params: List[mpz] = self._jls.protect(
                public_param=public_param,
                user_key=key,
                tau=current_round,
                x_u_tau=params,
                n_users=num_nodes
            )
        except (TypeError, ValueError) as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value} Error during parameter encryption. {exp}") from exp

        time_elapsed = time.process_time() - start
        logger.debug(f"Encryption of the parameters took {time_elapsed} seconds.")

        return [int(e_p) for e_p in encrypted_params]

    def aggregate(
            self,
            current_round: int,
            num_nodes: int,
            params: List[List[int]],
            key: int,
            biprime: int,
            total_sample_size: int,
            clipping_range: Union[int, None] = None
    ) -> List[float]:
        """Decrypt given parameters

        Args:
            current_round: The round that the aggregation will be done
            params: Aggregated/Summed encrypted parameters
            num_nodes: number of nodes
            key: The key that will be used for decryption
            biprime: Biprime number of `PublicParam`
            total_sample_size: sum of number of samples from all nodes
            clipping_range: Clipping range for reverse-quantization, should be the
                same clipping range used for quantization
        Returns:
            Aggregated parameters decrypted and structured

        Raises:
             FedbiomedSecaggCrypterError: bad parameters
             FedbiomedSecaggCrypterError: aggregation issue
        """
        start = time.process_time()

        if len(params) != num_nodes:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: Num of parameters that are received from nodes "
                f"does not match the number of nodes has been set for the encrypter. There might "
                f"be some nodes did not answered to training request or num of clients of "
                "`ParameterEncrypter` has not been set properly before train request.")

        if not isinstance(params, list) or not all([isinstance(p, list) for p in params]):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: The parameters to aggregate should "
                                              f"list containing list of parameters")

        if not all([all([isinstance(p_, int) for p_ in p]) for p in params]):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: Invalid parameter type. The parameters "
                                              f"should be type of integers.")

        # TODO provide dynamically created biprime. Biprime that is used
        #  on the node-side should matched the one used for decryption
        public_param = self._setup_public_param(biprime=biprime)
        key = ServerKey(public_param, key)

        params = self._convert_to_encrypted_number(params, public_param)

        try:
            sum_of_weights = self._jls.aggregate(
                sk_0=key,
                tau=current_round,  # The time period \\(\\tau\\)
                list_y_u_tau=params
            )
        except (ValueError, TypeError) as e:
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624.value}: The aggregation of encrypted parameters "
                                              f"is not successful: {e}")

        # TODO implement weighted averaging here or in `self._jls.aggregate`
        # Reverse quantize and division (averaging)
        logger.info(f"Aggregating {len(params)} parameters from {num_nodes} nodes.")
        aggregated_params = self._apply_average(sum_of_weights, total_sample_size)

        aggregated_params: List[float] = reverse_quantize(
            aggregated_params,
            clipping_range=clipping_range
        )
        time_elapsed = time.process_time() - start
        logger.debug(f"Aggregation is completed in {round(time_elapsed, ndigits=2)} seconds.")

        return aggregated_params

    @staticmethod
    def _apply_average(
            params: List[int],
            total_weight: int
    ) -> List:
        """Divides parameters with total weight

        Args:
            params: List of parameters
            total_weight: Total weight to divide
        
        Returns:
            List of averaged parameters
        """
        return divide(params, total_weight)

    @staticmethod
    def _apply_weighing(
            params: List[int],
            weight: int,
    ) -> List[int]:
        """Multiplies parameters with weight

        Args:
            params: List of parameters
            weight: Weight to multiply
        
        Returns:
            List of weighted parameters
        """
        return multiply(params, weight)
    
    @staticmethod
    def _convert_to_encrypted_number(
            params: List[List[int]],
            public_param: PublicParam
    ) -> List[List[EncryptedNumber]]:
        """Converts encrypted integers to `EncryptedNumber`

        Args:
            params: A list containing list of encrypted integers for each node
            public_param: Public parameter used while encrypting the model parameters
        Returns:
            list of `EncryptedNumber` objects
        """

        encrypted_number = []
        for parameters in params:
            encrypted_number.append([EncryptedNumber(public_param, mpz(param)) for param in parameters])

        return encrypted_number