# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from dataclasses import dataclass
from math import ceil
from Crypto.Cipher import ChaCha20
from typing import List, Optional, Dict

from fedbiomed.common.logger import logger
from ._base import EncrypterBase



class PRF(object):
    """Pseudo Random Function"""

    _zero = 0
    _nonce = _zero.to_bytes(12, "big")
    _element_size = 16
    security = 256

    def __init__(self, elementsize) -> None:
        super().__init__()

        # Not used
        self.bits_ptxt = elementsize

        # not used
        self.num_bytes = ceil(self._element_size / 8)

    def eval_key(self, key: bytes, t: int):
        """Evaluates the pairwise key

        Args:
            key: key as bytes
            t: Period of time
        """
        time_period = t.to_bytes(self._element_size, 'big')
        c = ChaCha20.new(key=key, nonce=PRF._nonce).encrypt(time_period)

        # the output is a 16 bytes string, pad it to 32 bytes
        # TODO fix it, I don't know if it is correct to pad with zeros
        c = c + b'\x00' * 16
        return c

    def eval_vector(self, seed, vector_size):

        c = ChaCha20.new(key=seed, nonce=PRF._nonce)
        data = b"secr" * vector_size

        return c.encrypt(data)


class Flamingo(EncrypterBase):
    """
    """

    prf: Optional[PRF] = None
    _vector_dtype: str = 'uint32'

    def __init__(
        self,
        self_identifier: str,
        parties: List[str],
        secrets: Dict[str, bytes]
    ) -> None:
        """Constructs Flamingo secure aggregation crypter

        Args:
            self_identifies: An id or identifier string that corresponds to the party
                that runs protection
            parties: Party ids or identifiers (same type of identifier as `self_identifier`)
                that participates secure ggregation except server.
        """

        self._self_identifier = self_identifier
        self._parties = parties
        self._secrets = secrets


    def setup_pairwise_secrets(self, my_node_id:int, nodes_ids: List[str], num_params: int) -> None:
        """"""
        Flamingo.prf = PRF(vectorsize=num_params, elementsize=16)
        self.pairwise_secrets = {}
        self.pairwise_seeds = {}
        self.my_node_id = str(my_node_id)
        # this is just an hardcoded example
        for node_id in nodes_ids:
            if node_id != my_node_id:
                # create 32 bytes secret of zeros
                self.pairwise_secrets[node_id] = b'\x02' * 32


    def protect(
            self,
            t: int,
            vector: List[int],
    ) -> List[int]:

        """
        TODO: Add docstring
        """
        params = np.array(vector, dtype=self.vector_dtype)
        vec = np.zeros(len(params), dtype=self.vector_dtype)

        # Pseudo random function
        prf = PRF()

        for party_id in self._parties:
            if party_id == self._self_identifier:
                continue

            secret = self.pairwise_secrets[party_id]
            # generate seed for pairwise encryption
            pairwise_seed = prf.eval_key(key=secret, t=t)
            # expand seed to a random vector
            pairwise_vector = prf.eval_vector(seed=pairwise_seed, vector_size=len(vector))
            pairwise_vector = np.frombuffer(pairwise_vector, dtype=self._vector_dtype)

            if party_id < self._self_identifier:
                vec += pairwise_vector
            else:
                vec -= pairwise_vector

        encrypted_params = vec + params
        return encrypted_params.tolist()

    def aggregate(
            self,
            vectors: List[List[int]],
    ) -> List[int]:
        """Aggregates encrypted vectors

        Args:
            vectors: Protected vectors

        Returns:
            Final aggregated vector
        """
        vectors = np.array(vectors, dtype=self.vector_dtype)
        sum_params = np.sum(vectors, axis=0)
        sum_params = sum_params.astype(np.int32)
        sum_params = sum_params.tolist()

        return sum_params
