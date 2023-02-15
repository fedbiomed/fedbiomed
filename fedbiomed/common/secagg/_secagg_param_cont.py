import numpy as np


from math import ceil, log2
from typing import List, Union
from tinydb import TinyDB
from gmpy2 import mpz


from fedbiomed.common.exceptions import FedbiomedEncryptionError


from ._jls import JLS, EncryptedNumber as EN, ServerKey, UserKey, FDH, DEFAULT_KEY_SIZE, PublicParam
from ._jls_utils import quantize, reverse_quantize
from ._vector_encoding import VES


DEFAULT_CLIPPING: int = 3
"""
Default clipping value that is going to be used to quantize list of parameters 
"""



class VEParameters:
    """Constants class for vector encoder parameters"""

    KEY_SIZE: int = 2048
    NUM_CLIENTS: int = 2
    VALUE_SIZE: int = ceil(log2(10000))
    VECTOR_SIZE: int = 1199882


class EncryptedNumber(EN):
    """Extends EncryptedNumber class to be able to `sum` functionality"""

    def __radd__(self, value: Union[EN, mpz]):
        """Allows summing parameters using built-in `sum` method

        Args:
            value: Value to add. It can be an instance of `mpz` or EncryptedNumber
        """
        return super().__add__(value)


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
        self._num_nodes = None
        self._vector_encoder = VES(
            ptsize=VEParameters.KEY_SIZE // 2,
            addops=None,
            valuesize=VEParameters.VALUE_SIZE,
            vectorsize=VEParameters.VECTOR_SIZE,
        )
        self._jls = JLS(
            nusers=None,
            VE=self._vector_encoder
        )

    def set_num_nodes(
            self,
            num_clients: int
    ) -> int:
        """Sets number of clients for vector encoder

        Number of clients/nodes except researcher (aggregator) should be dynamic where
        it may change from one round to other or one experiment to other.

        Args:
            num_clients: Number of clients that participates in the training

        Returns:
            number of clients that is set in vector encoder
        """
        self._num_nodes = num_clients

        # Update vector encoder
        self._vector_encoder.addops = num_clients
        self._vector_encoder.elementsize = self._vector_encoder.valuesize + ceil(log2(num_clients))

        # Update JLS
        # FIXME: currently `nusers` attribute has impact on nothing. It was only used for calculating
        #  SerkeyKey and UserKey before
        self._jls.nusers = num_clients

        return self._num_clients

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
        if isinstance(key, int):
            raise FedbiomedEncryptionError(f"The argument key must be integer")

        # Number of nodes should be set for vector encoder
        self.set_num_nodes(num_nodes)

        clipping_range = self._get_clipping_value(params)

        weights = quantize(
            weight=params,
            clipping_range=clipping_range
        ).tolist()

        public_param = self._setup_public_param()

        # Instantiates UserKey object
        key = UserKey(public_param, key)

        # Encrypt parameters
        encrypted_params: EncryptedNumber = self._jls.Protect(
            pp=public_param,
            sk_u=key,
            tau=current_round,
            x_u_tau=weights,
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

    def decrypt(self, current_round, params, key: int):
        """Decrypt given parameters

        Args:
            current_round: The round that the aggregation will be done
            params: Aggregated/Summed encrypted parameters
            key: The key that will be used for decryption

        """

        num_nodes = len(params)

        self.set_num_nodes(num_nodes)

        # TODO: This check should be done before executing this method
        # if len(params) != self._num_nodes:
        #     raise FedbiomedEncryptionError(f"Num of parameters that are received from node does not match the num of "
        #                                    f"nodes has been set for the encrypter. There might be some nodes did "
        #                                    f"not answered to training request or num of clients of "
        #                                    f"`ParameterEncrypter` has not been set properly before train request.")

        # TODO provide dynamically created biprime. Biprime that is used
        #  on the node-side should matched the one used for decryption
        public_param = self._setup_public_param()

        key = ServerKey(public_param, key)

        params = key.decrypt(params, current_round)

        sum_of_weights = self._vector_encoder.decode(params)

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
            return SecaggCrypter._get_clipping_value(params, clipping+1)
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
