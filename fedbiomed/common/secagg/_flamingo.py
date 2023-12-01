# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Dict

import numpy as np

from math import ceil

from Crypto.Cipher import ChaCha20


from fedbiomed.common.logger import logger



class PRF(object):
    _zero = 0
    _nonce = _zero.to_bytes(12, "big")
    security = 256

    def __init__(self, vectorsize, elementsize) -> None:
        super().__init__()
        self.vectorsize = vectorsize
        self.bits_ptxt = elementsize
        self.num_bytes = ceil(elementsize / 8)

    def eval_key(self, key: bytes, round: int):
        round_number_bytes = round.to_bytes(16, 'big')
        c = ChaCha20.new(key=key, nonce=PRF._nonce).encrypt(round_number_bytes)
        # the output is a 16 bytes string, pad it to 32 bytes
        # TODO fix it, I don't know if it is correct to pad with zeros
        c = c + b'\x00' * 16
        return c

    def eval_vector(self, seed):
        c = ChaCha20.new(key=seed, nonce=PRF._nonce)
        data = b"secr" * self.vectorsize
        return c.encrypt(data)


class Flamingo:
    """
    TODO: Add docstring
    """
    prf: Optional[PRF] = None
    def __init__(self) -> None:
        super().__init__()
        self.vector_dtype = 'uint32'

    def setup_pairwise_secrets(self, my_node_id:int, nodes_ids: List[str], num_params: int) -> None:
        """
        TODO: Add docstring
        """
        Flamingo.prf = PRF(vectorsize=num_params, elementsize=16)
        self.pairwise_secrets = {}
        self.pairwise_seeds = {}
        self.my_node_id = str(my_node_id)
        # this is just an hardcoded example
        for node_id in nodes_ids:
            if node_id != my_node_id:
                #create 32 bytes secret of zeros
                self.pairwise_secrets[node_id] = b'\x02' * 32


    def protect(
            self,
            current_round: int,
            params: List[int], 
            node_ids,
            )-> List[int]:
        """
        TODO: Add docstring
        """
        params = np.array(params, dtype=self.vector_dtype)
        vec = np.zeros(len(params), dtype=self.vector_dtype)
        for node_id in node_ids:
            if node_id == self.my_node_id:
                continue
            secret = self.pairwise_secrets[node_id]
            # generate seed for pairwise encryption
            pairwise_seed = Flamingo.prf.eval_key(key=secret, round=current_round)
            # expand seed to a random vector
            pairwise_vector = Flamingo.prf.eval_vector(seed=pairwise_seed)
            pairwise_vector = np.frombuffer(pairwise_vector, dtype=self.vector_dtype)
            if node_id < self.my_node_id:
                vec += pairwise_vector
            else:
                vec -= pairwise_vector

        encrypted_params = vec + params
        return encrypted_params.tolist()

    def aggregate(
            self,
            params: Dict[str, np.ndarray],
    ) -> List[int]:
        """
        TODO: Add docstring
        """
        params = np.array(params, dtype=self.vector_dtype)
        sum_params = np.sum(params, axis=0)
        sum_params = sum_params.astype(np.int32)
        sum_params = sum_params.tolist()

        return sum_params
