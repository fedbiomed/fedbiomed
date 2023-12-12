
import time

from typing import List, Optional
from gmpy2 import mpz

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers, VEParameters

from ._baase import EncrypterBase
from .utils import divide, multiply, quantize, reverse_quantize


class SecaggAlgorithms:
    """Supported secure aggregation algorithms"""

    JLS: str  = 'Joye-Libert'
    FLAMINGO: str = 'Flamingo'


class Crypter:


    _crypter: EncrypterBase

    def __init__(
        self,
        crypter: EncrypterBase
    ) -> None:
        """Constructs crypter"""

        self._crypter = crypter

    def encrypt(
        self,
        round_: int,
        vector: List[int],
        parties: List[str],
        clipping_range: int = VEParameters.CLIPPING_RANGE,

    ) -> List[int]:
        """Encrypts/protects given vector with secure aggregation algorithm

        Args:
            round_: Round number of the training or any iterative operation
            vector: Vector to encrpyted/protect
            parties: Participating parties to the secure aggregation
            clipping_range: range to use for qauntization floats

        Returns:
            Encrypted list ofbig integers
        """

        start = time.process_time()
        vector = quantize(weights=vector, clipping_range=clipping_range)

        encrypted_vector: List[int] = self._crypter.protect(
            tau=round_,
            x_u_tau=vector,
            n_users=len(parties)
        )

        time_elapsed = time.process_time() - start
        logger.debug(f"Encryption of the parameters took {time_elapsed} seconds.")

        return encrypted_vector


    def aggregate(
        self,
        round_: int,
        vectors: List[List[int]],
        weight_deminator: Optional[int] = None,
        clipping_range: Optional[int] = None
    ) -> List[float]:
        """Aggregates encrypted vectors

        Args:
            round_: Current round of iterative operation
            vectors: List of encrypted vectors
            weight_deminator: Weighing deminator if it is applied before encryption
                for each encrpted vector
            clipping_range: Cliping range for reverse quantizaztion. It should be
                eqaul to the one used for ecryption
        """
        start = time.process_time()
        vector_sum = self._crypter.aggregate(
            t=round_,
            vectors=vectors
        )

        if weight_deminator:
            vector_sum = self._apply_average(vector_sum, weight_deminator)

        aggregated_vector: List[float] = reverse_quantize(
            vector_sum,
            clipping_range=clipping_range)

        time_elapsed = time.process_time() - start
        logger.debug(f"Aggregation is completed in {round(time_elapsed, ndigits=2)} seconds.")

        return aggregated_vector



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
