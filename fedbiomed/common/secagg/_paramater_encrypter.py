import numpy as np


from math import ceil, log2
from typing import List
from tinydb import TinyDB
from gmpy2 import mpz


from fedbiomed.common.exceptions import FedbiomedEncryptionError


from ._jls import JLS, EncryptedNumber, ServerKey, UserKey, FDH, DEFAULT_KEY_SIZE, PublicParam
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


class ParameterEncrypter:
    """


    Attributes:
        _num_clients: Number of clients participating federated learning experiment/round
        _vector_encoder: Encodes given parameters vector
        _jls: Joye-Libert homomorphic encryption class
    """
    def __init__(self, db: str) -> None:
        """Constructs ParameterEncrypter

        """
        self._num_clients = None
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


    def set_num_clients(
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

        self._vector_encoder.addops = num_clients
        self._jls.nusers = num_clients

        return self._vector_encoder.addops

    @staticmethod
    def _setup_public_param() -> PublicParam:
        """Creates public parameter for encryption

        Returns:
            Public parameters
        """

        key_size = VEParameters.KEY_SIZE

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
            current_round: int,
            params: List,
            key: int
    ) -> dict:
        """Encrypts model parameters.


        Args:
            current_round:
            params: List of flatten parameters
            key: Key to encrypt
        """

        # Make use the key is instance of
        if isinstance(key, int):
            raise FedbiomedEncryptionError(f"The argument key must be integer")

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

    def decrypt(self, **kwargs):

        # TODO provide dynamicly created birpime
        public_param = self._setup_public_param()


        pass

    @staticmethod
    def _get_clipping_value(params: List) -> int:
        """Gets minimum clipping value by checking minimum value of the params list.

        Args:
            params: List of parameters (vector) that are going to be encrypted.

        Returns
            Clipping value for quantizing.
        """


        min_val = min(params)
        if min_val < DEFAULT_CLIPPING:
            return round(min + 1)
        else:
            return DEFAULT_CLIPPING
