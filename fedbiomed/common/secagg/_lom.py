import math
import secrets
from typing import List, Dict
from multiprocessing import Pool, cpu_count

import numpy as np

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.backends import default_backend


from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedSecaggError

_MAX_ROUND=1000


class PRF:
    """
    Pseudorandom Function (PRF) class using the ChaCha20 stream cipher.

    Attributes:
        nonce (bytes): A 12-byte nonce used for encryption.
    """
    def __init__(self, nonce: bytes) -> None:
        self._nonce = nonce
        # FIXME: check that nonce is a 12 bytes value

    def eval_key(self, pairwise_secret: bytes, tau: int) -> bytes:
        """
        Evaluates a pseudorandom key for a given round.

        Args:
            pairwise_secret (bytes): A secret key shared between nodes.
            round (int): The current round number.

        Returns:
            bytes: A 32-byte pseudorandom key.
        """
        tau = tau.to_bytes(16, 'big')
        try:
            encryptor = Cipher(
                algorithms.ChaCha20(pairwise_secret, self._nonce),
                mode=None,
                backend=default_backend()
            ).encryptor()
        except ValueError as ve:
            raise FedbiomedSecaggError(f"{ErrorNumbers.FB417.value}: Error while ciphering: got exception {ve}")
        c = encryptor.update(tau) + encryptor.finalize()
        # the output is a 16 bytes string, pad it to 32 bytes
        c = c + b'\x00' * 16

        return c

    def eval_vector(self, seed: bytes, tau: int, input_size: int) -> bytes:
        """
        Evaluates a pseudorandom vector based on the seed.

        Args:
            seed (bytes): A 32-byte seed for generating the vector.
            tau (int): The current round number.
            input_size (int): The size of the input vector.

        Returns:
            bytes: A pseudorandom vector of the specified size.
        """
        encryptor = Cipher(
            algorithms.ChaCha20(seed, self._nonce),
            mode=None,
            backend=default_backend()
        ).encryptor()

        # TODO: Better handling limits for secure aggregation
        if not (input_size + _MAX_ROUND) <= 2**32:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB417.value}: Can not perform encryiton due to large input vector. input_size "
                f"({input_size}) + MAX_ROUND ({_MAX_ROUND}) allowed is greater than 2**32"
            )

        # create a list of indices from 0 to input_size where each element is concatenated with tau
        taus = b''.join([(i + tau).to_bytes(4, 'big') for i in range(input_size)])
        return encryptor.update(taus) + encryptor.finalize()


class LOM:
    """
    Lightweight Obfuscation Mechanism (LOM) class for protecting and aggregating data.

    Attributes:
        _prf: An instance of the PRF class.
        _vector_dtype: The data type of the vector.
    """
    def __init__(
        self,
        nonce: bytes | None = None
    ) -> None:

        if not nonce:
            nonce = secrets.token_bytes(16)

        self._prf: PRF = PRF(nonce)
        self._vector_dtype: str = 'uint32'
        self._values_bit = np.iinfo(np.dtype(self._vector_dtype)).bits  # should be equal to 32 bit


    def protect(
        self,
        node_id: str,
        pairwise_secrets: Dict[str, bytes],
        tau: int,
        x_u_tau: List[int],
        node_ids: List[str]
    ) -> List[int]:
        """
        Protects the input vector by applying a mask based on pairwise secrets.

        Args:
            node_id: Id of the node that applies encryption
            pairwise_secrets: DH agreed secrets between node that applies encryption and others
            tau: The current round number.
            x_u_tau: The input vector to be protected.
            node_ids: A list of node IDs participates aggregation.

        Raises:
            FedBioMedError: raises if the input vector `x_u_tau` contains any 
                values that exceed  `32 - log_2(numner_of_nodes)`, where `32` is 
                the number of bit for each value (`uint32`). Not respecting the above condition 
                can lead to computation overflow.

        Returns:
            The protected (masked) vector.
        """

        num_nodes = len(node_ids)
        _max_bit_length = self._values_bit - math.log2(num_nodes)
        # FIXME: is a ceiling missing in this equation (ceil(math.log2(num_nodes)))
        if any(val.bit_length() > _max_bit_length for val in x_u_tau):
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB417.value}: Bit length of one or more values of input vector has more bits "
                f"that {_max_bit_length}. This could lead to computation overflow"
            )

        x_u_tau = np.array(x_u_tau, dtype=self._vector_dtype)
        mask = np.zeros(len(x_u_tau), dtype=self._vector_dtype)
        for pair_id in node_ids:

            if pair_id == node_id:
                continue

            secret = pairwise_secrets[pair_id]

            pairwise_seed = self._prf.eval_key(
                pairwise_secret=secret,
                tau=tau)

            # print(len(pairwise_seed))
            pairwise_vector = self._prf.eval_vector(
                seed=pairwise_seed,
                tau=tau,
                input_size=len(x_u_tau))

            pairwise_vector = np.frombuffer(pairwise_vector, dtype=self._vector_dtype)

            if pair_id < node_id:
                mask += pairwise_vector
            else:
                mask -= pairwise_vector

        encrypted_params = mask + x_u_tau

        return encrypted_params.tolist()

    def aggregate(self, list_y_u_tau: List[int]) -> List[int]:
        """
        Aggregates multiple vectors into a single vector.

        Args:
            list_y_u_tau: A dictionary of vectors from different nodes.

        Returns:
            The aggregated vector.
        """
        list_y_u_tau = np.array(list_y_u_tau, dtype=self._vector_dtype)
        decrypted_vector = np.sum(list_y_u_tau, axis=0)
        decrypted_vector = decrypted_vector.astype(np.int32)
        decrypted_vector = decrypted_vector.tolist()

        return decrypted_vector
