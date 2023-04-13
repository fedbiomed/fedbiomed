# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


import time

from typing import List
from gmpy2 import mpz
import numpy as np

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
    reverse_quantize



"""
Default clipping value that is going to be used to quantize list of parameters 
"""


class SecaggCrypter:
    """Secure aggregation encryption and decryiton manager.

    This class is responsible for encrypting model parameters using Joye-Libert secure
    aggregation scheme. It also aggregates encrypted model parameters and decrypts
    to retrieve final model parameters as vector. This vector can be loaded into model
    by converting it proper format for the framework.
    """

    def __init__(self) -> None:
        """Constructs ParameterEncrypter"""
        self._jls = JoyeLibert()

    @staticmethod
    def _setup_public_param() -> PublicParam:
        """Creates public parameter for encryption

        Returns:
            Public parameters
        """

        key_size = VEParameters.KEY_SIZE

        # TODO: Used hard-coded/pre-saved Biprime
        p = mpz(
            7801876574383880214548650574033350741129913580793719706746361606042541080141291132224899113047934760791108387050756752894517232516965892712015132079112571
        )
        q = mpz(
            7755946847853454424709929267431997195175500554762787715247111385596652741022399320865688002114973453057088521173384791077635017567166681500095602864712097
        )

        n = p * q
        fdh = FDH(key_size, n * n)

        return PublicParam(n_modulus=n,
                           bits=key_size // 2,
                           hashing_function=fdh.H)

    def encrypt(
            self,
            num_nodes: int,
            current_round: int,
            params: List[float],
            key: int,
            weight: int = None
    ) -> List[int]:
        """Encrypts model parameters.

        Args:
            num_nodes: Number of nodes that is expected to encrypt parameters for aggregation
            current_round: Current round of federated training
            params: List of flatten parameters
            key: Key to encrypt
            weight: Weight for the params

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

        params = self.apply_weighing(params, weight)

        params = quantize(weights=params).tolist()
        public_param = self._setup_public_param()

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
            total_sample_size: int
    ) -> np.ndarray:
        """Decrypt given parameters

        Args:
            current_round: The round that the aggregation will be done
            params: Aggregated/Summed encrypted parameters
            num_nodes: number of nodes
            key: The key that will be used for decryption
            total_sample_size: sum of number of samples from all nodes

        Returns:
            Aggregated parameters decrypted and structured

        Raises:
             FedbiomedSecaggCrypterError: bad parameters
             FedbiomedSecaggCrypterError: aggregation issue
        """
        start = time.process_time()

        if len(params) != num_nodes:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB624.value}Num of parameters that are received from node "
                f"does not match the num of nodes has been set for the encrypter. There might "
                f"be some nodes did not answered to training request or num of clients of "
                "`ParameterEncrypter` has not been set properly before train request.")

        if not isinstance(params, list) or not all([isinstance(p, list) for p in params]):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: The parameters to aggregate should "
                                              f"list containing list of parameters")

        if not all([all([isinstance(p_, int) for p_ in p]) for p in params]):
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB624}: Invalid parameter type. The parameters "
                                              f"should be type of integers.")

        params = self._convert_to_encrypted_number(params)

        # TODO provide dynamically created biprime. Biprime that is used
        #  on the node-side should matched the one used for decryption
        public_param = self._setup_public_param()
        key = ServerKey(public_param, key)

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
        aggregated_params = reverse_quantize(
            self.apply_average(sum_of_weights, num_nodes, total_sample_size)
        ).tolist()

        time_elapsed = time.process_time() - start
        logger.debug(f"Secure aggregation took {time_elapsed} seconds.")

        return aggregated_params

    @staticmethod
    def apply_average(
            params: List[int],
            num_nodes: int,
            total_sample_size: int
    ) -> List:
        """Takes the average of summed quantized parameters.

        Args:
            params: List of aggregated/summed parameters
            num_nodes: Number of nodes participated in the training
            total_sample_size: Num of total samples used for federated training

        Returns:
            Averaged parameters
        """

        return [param / num_nodes for param in params]

    @staticmethod
    def apply_weighing(
            params: List[int],
            weight: int,
    ) -> List[int]:
        """Takes the average of summed parameters.

        Args:
            params: A list containing list of parameters
            weight: The weight factor to apply

        Returns:
            Weighed parameters
        """

        # TODO: Currently weighing is not activated due to CLIPPING_RANGE problem.
        #  Implement weighing.
        return [param * 1 for param in params]

    def _convert_to_encrypted_number(self, params: List[List[int]]) -> List[List[EncryptedNumber]]:
        """Converts encrypted integers to `EncryptedNumber`

        Args:
            params: A list containing list of encrypted integers for each node

        Returns:
            list of `EncryptedNumber` objects
        """

        # Set public params
        public_param = self._setup_public_param()

        encrypted_number = []
        for parameters in params:
            encrypted_number.append([EncryptedNumber(public_param, mpz(param)) for param in parameters])

        return encrypted_number
