# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


import time
import random

from typing import Dict, List, Union, Optional
from gmpy2 import mpz

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers, SAParameters
from fedbiomed.common.logger import logger

from fedbiomed.common.utils import quantize, \
    reverse_quantize, \
    multiply, \
    divide

from ._jls import JoyeLibert, \
    EncryptedNumber, \
    ServerKey, \
    UserKey, \
    FDH, \
    PublicParam

from ._lom import LOM


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

        key_size = SAParameters.KEY_SIZE
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
            weight: Optional[int] = None,
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

        # We multiply the parameters with the weight, and we get params in
        # the range [0, 2^(log2(VEParameters.TARGET_RANGE) + log2(VEParameters.WEIGHT_RANGE)) - 1]
        # Check if weight if num_bits of weight is less than VEParameters.WEIGHT_RANGE
        if weight is not None:
            if 2**weight.bit_length() > SAParameters.WEIGHT_RANGE:
                raise FedbiomedSecaggCrypterError(
                    f"{ErrorNumbers.FB624.value}: The weight is too large. The weight should be less than "
                    f"{SAParameters.WEIGHT_RANGE}, but got {weight}"
                )
            params = self._apply_weighting(params, weight)


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
            clipping_range: Union[int, None] = None,
            num_expected_params: int = 1
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
            num_expected_params: number of parameters to decode from the `params`
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

        if not isinstance(params, list) or not all(isinstance(p, list) for p in params):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: The parameters to aggregate should be a "
                                              f"list containing list of parameters")

        if not all(all(isinstance(p_, int) for p_ in p) for p in params):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: Invalid parameter type. The parameters "
                                              f"should be of type of integers.")

        # TODO provide dynamically created biprime. Biprime that is used
        #  on the node-side should matched the one used for decryption
        public_param = self._setup_public_param(biprime=biprime)
        key = ServerKey(public_param, key)

        params = self._convert_to_encrypted_number(params, public_param)

        try:
            sum_of_weights = self._jls.aggregate(
                sk_0=key,
                tau=current_round,  # The time period \\(\\tau\\)
                list_y_u_tau=params,
                num_expected_params=num_expected_params
            )
        except (ValueError, TypeError) as e:
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624.value}: The aggregation of encrypted parameters "
                                              f"is not successful: {e}")

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
        # Check that quantized model weights are unsigned integers, for robustness sake
        if any([v < 0 for v in params]):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: Cannot compute weighted average, values outside of bounds")

        return divide(params, total_weight)

    @staticmethod
    def _apply_weighting(
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
        m = multiply(params, weight)

        # Check that quantized model weights are in the correct range, for robustness sake
        max_val = SAParameters.TARGET_RANGE - 1
        if any([v > max_val or v < 0 for v in params]):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: Cannot apply weight to parameters, values outside of bounds"
            )

        return m

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


class SecaggLomCrypter(SecaggCrypter):
    """Low-Overhead Masking secure aggregation controller"""


    def __init__(
        self,
        nonce: str | None = None
    ):
        """LOM Secure aggregation to encrypt and aggregate

        Args:
            nonce: `nonce` to use in encryption. Needs to be the same between the parties of
                the LOM computation. Can be disclosed (public). Must not be re-used.
        """
        if nonce:
            # The security relies on the non-reuse of the nonce.
            # We also need to ensure 128 bits
            # Padding is enough, using `random()` is misleading (no additional security)
            nonce = str.encode(nonce).zfill(16)[:16]

        self._lom = LOM(nonce)


    def encrypt(
        self,
        current_round: int,
        node_id: str,
        params: List[float],
        pairwise_secrets: Dict[str, bytes],
        node_ids: List[str],
        clipping_range: Union[int, None] = None,
        weight: Optional[int] = None,
    ) -> List[int]:
        """Encrypts model parameters.

        Args:
            current_round: Current round of federated training
            node_id: ID of the node applies encryption
            params: List of flatten parameters
            pairwise_secrets: DH agreed secrets between node that applies encryption and others
            node_ids: All nodes that participates secure aggregation
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

        if not all(isinstance(p, float) for p in params):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: The parameters to encrypt should list of floats. "
                f"There are one or more than a value that is not type of float."
            )

        params = quantize(weights=params,
                          clipping_range=clipping_range)

        if weight is not None:
            if 2**weight.bit_length() > SAParameters.WEIGHT_RANGE:
                raise FedbiomedSecaggCrypterError(
                    f"{ErrorNumbers.FB624.value}: The weight is too large. The weight should be less than "
                    f"{SAParameters.WEIGHT_RANGE}."
                )
            params = self._apply_weighting(params, weight)

        try:
            # Encrypt parameters
            encrypted_params: List[int] = self._lom.protect(
                pairwise_secrets=pairwise_secrets,
                node_id=node_id,
                tau=current_round,
                x_u_tau=params,
                node_ids=node_ids
            )
        except (TypeError, ValueError) as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value} Error during parameter encryption. {exp}") from exp


        time_elapsed = time.process_time() - start
        logger.debug(f"Encryption of the parameters took {time_elapsed} seconds.")

        return encrypted_params

    def aggregate(
            self,
            params: List[List[int]],
            total_sample_size: int,
            clipping_range: Union[int, None] = None,
    ) -> List[float]:
        """Decrypt given parameters

        Args:
            params: Aggregated/Summed encrypted parameters
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

        if not isinstance(params, list) or not all(isinstance(p, list) for p in params):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: The parameters to aggregate should be a "
                                              f"list containing list of parameters")

        if not all(all(isinstance(p_, int) for p_ in p) for p in params):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: Invalid parameter type. The parameters "
                                              f"should be of type of integers.")

        num_nodes = len(params)

        try:
            sum_of_weights = self._lom.aggregate(
                list_y_u_tau=params,
            )
        except (ValueError, TypeError) as e:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}: The aggregation of encrypted parameters "
                f"is not successful: {e}") from e


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


