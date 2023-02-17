from typing import List, Union
from gmpy2 import mpz

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers

from ._jls import JoyeLibert, EncryptedNumber, ServerKey, UserKey, FDH, PublicParam
from ._jls_utils import quantize, reverse_quantize

DEFAULT_CLIPPING: int = 3
"""
Default clipping value that is going to be used to quantize list of parameters 
"""


class SecaggCrypter:
    """

    Attributes:
        _num_clients: Number of clients participating federated learning experiment/round
        _vector_encoder: Encodes given parameters vector
        _jls: Joye-Libert homomorphic encryption class
    """

    def __init__(self) -> None:
        """Constructs ParameterEncrypter

        """

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

        return PublicParam(n, key_size // 2, fdh.H)

    def encrypt(
            self,
            num_nodes: int,
            current_round: int,
            params: List[float],
            key: int
    ) -> List[int]:
        """Encrypts model parameters.


        Args:
            num_nodes: Number of nodes that is expected to encrypt parameters for aggregation
            current_round: Current round of federated training
            params: List of flatten parameters
            key: Key to encrypt
        """

        # Make use the key is instance of
        if not isinstance(key, int):
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB620.value}: The argument `key` must be integer"
            )

        clipping_range = self._get_clipping_value(params)

        weights = quantize(
            weight=params,
            clipping_range=clipping_range
        ).tolist()

        public_param = self._setup_public_param()

        # Instantiates UserKey object
        key = UserKey(public_param, key)

        # Encrypt parameters
        encrypted_params: List[EncryptedNumber] = self._jls.protect(
            public_param=public_param,
            sk_u=key,
            tau=current_round,
            x_u_tau=weights,
            n_users=num_nodes
        )

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
            key: int):
        """Decrypt given parameters

        Args:
            current_round: The round that the aggregation will be done
            params: Aggregated/Summed encrypted parameters
            num_nodes:
            key: The key that will be used for decryption

        """

        if len(params) != self._num_nodes:
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
                round_=current_round,  # The time period \\(\\tau\\)
                list_y_u_tau=params
            )
        except (ValueError, TypeError) as exp:
            raise FedbiomedSecaggCrypterError(f"The aggregation of encrypted parameters is not "
                                              f"successful: {exp}") from exp

        # TODO implement weighted everaging here or in `self._jls.aggregate`
        # Reverse quantize and division (averaging)
        aggregated_params = reverse_quantize(
            self.quantized_divide(sum_of_weights, num_nodes)
        ).tolist()

        return aggregated_params

    @staticmethod
    def _get_clipping_value(params: List, clipping: int = DEFAULT_CLIPPING) -> int:
        """Gets minimum clipping value by checking minimum value of the params list.

        Args:
            params: List of parameters (vector) that are going to be encrypted.

        Returns
            Clipping value for quantizing.
        """

        min_val = min(params)
        max_val = max(params)

        if min_val < -clipping or max_val > clipping:
            return SecaggCrypter._get_clipping_value(params, clipping + 1)
        else:
            return clipping

    @staticmethod
    def quantized_divide(params: List, num_clients: int) -> List:
        """Average

        Args:
            params: List of aggregated/summed parameters
            num_clients: Number of clients/nodes

        Returns:
            Averaged parameters
        """
        return [param / num_clients for param in params]

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
