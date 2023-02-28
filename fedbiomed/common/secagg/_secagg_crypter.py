import time

from typing import List
from gmpy2 import mpz

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers

from ._jls import JoyeLibert, EncryptedNumber, ServerKey, UserKey, FDH, PublicParam, VEParameters
from ._jls_utils import quantize, reverse_quantize, CLIPPING_RANGE
from ..logger import logger

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
        """

        start = time.process_time()

        # Make use the key is instance of
        if not isinstance(key, int):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB620.value}: The argument `key` must be integer"
            )

        if weight is not None:
            params = [param * 1 for param in params]

        params = quantize(weights=params).tolist()
        public_param = self._setup_public_param()

        # Instantiates UserKey object
        key = UserKey(public_param, key)

        # Encrypt parameters
        encrypted_params: List[EncryptedNumber] = self._jls.protect(
            public_param=public_param,
            sk_u=key,
            tau=current_round,
            x_u_tau=params,
            n_users=num_nodes
        )

        time_elapsed = time.process_time() - start
        logger.debug(f"Encryption of the parameters took {time_elapsed} seconds.")

        return self.convert_encrypted_number_to_int(encrypted_params)

    @staticmethod
    def convert_encrypted_number_to_int(
            params: List[EncryptedNumber]
    ) -> List[int]:
        """Converts given `EncryptedNumber` to integer

        Args:
            params: Encrypted model parameters

        Returns:
            List of quantized and encrypted model parameters
        """

        return list(map(lambda encrypted_number: int(encrypted_number.ciphertext), params))

    def aggregate(
            self,
            current_round: int,
            num_nodes: int,
            params: List[List[EncryptedNumber]],
            key: int,
            total_sample_size: int
    ):
        """Decrypt given parameters

        Args:
            current_round: The round that the aggregation will be done
            params: Aggregated/Summed encrypted parameters
            num_nodes:
            key: The key that will be used for decryption
            total_sample_size:

        """
        start = time.process_time()

        if len(params) != num_nodes:
            raise FedbiomedSecaggCrypterError(
                f"Num of parameters that are received from node does not match the num of "
                f"nodes has been set for the encrypter. There might be some nodes did "
                f"not answered to training request or num of clients of "
                f"`ParameterEncrypter` has not been set properly before train request.")

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
            raise FedbiomedSecaggCrypterError(f"The aggregation of encrypted parameters is not "
                                              f"successful: {e}")

        # TODO implement weighted averaging here or in `self._jls.aggregate`
        # Reverse quantize and division (averaging)
        aggregated_params = reverse_quantize(
            self.quantized_divide(sum_of_weights, num_nodes, total_sample_size)
        ).tolist()

        time_elapsed = time.process_time() - start
        logger.debug(f"Secure aggregation took {time_elapsed} seconds.")

        return aggregated_params

    @staticmethod
    def _get_clipping_value(params: List, clipping: int = CLIPPING_RANGE) -> int:
        """Gets minimum clipping value by checking minimum value of the params list.

        Args:
            params: List of parameters (vector) that are going to be encrypted.

        Returns
            Clipping value for quantization.
        """

        min_val = min(params)
        max_val = max(params)

        if min_val < -clipping or max_val > clipping:
            return SecaggCrypter._get_clipping_value(params, clipping + 1)
        else:
            return clipping

    @staticmethod
    def quantized_divide(
            params: List,
            num_nodes: int,
            total_sample_size: int
    ) -> List:
        """Average

        Args:
            params: List of aggregated/summed parameters
            total_sample_size: Num of total samples used for federated training

        Returns:
            Averaged parameters
        """
        #TODO: Use total sample size for weighted averaging

        return [param / num_nodes for param in params]

    def convert_to_encrypted_number(self, params: List[List[int]]) -> List[List[EncryptedNumber]]:
        """Converts encrypted integers to EncryptedNumber

        Args:
            params: A list containing list of encrypted integers for each node
        """

        # Set public params
        public_param = self._setup_public_param()

        encrypted_number = []
        for parameters in params:
            encrypted_number.append([EncryptedNumber(public_param, mpz(param)) for param in parameters])

        return encrypted_number
