from typing import List, Dict
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend

class PRF:
    """
    Pseudorandom Function (PRF) class using the ChaCha20 stream cipher.

    Attributes:
        nonce (bytes): A 12-byte nonce used for encryption.
    """
    def __init__(self, nonce: bytes) -> None:
        self._nonce = nonce

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
        encryptor = Cipher(
            algorithms.ChaCha20(pairwise_secret, self._nonce),
            mode=None,
            backend=default_backend()
        ).encryptor()
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
        # create a list of indices from 0 to input_size where each element is concatenated with tau
        taus = b''.join([i.to_bytes(2, 'big') + tau.to_bytes(2, 'big') for i in range(input_size)])
        return encryptor.update(taus) + encryptor.finalize()


class LOM:
    """
    Lightweight Obfuscation Mechanism (LOM) class for protecting and aggregating data.

    Attributes:
        prf (PRF): An instance of the PRF class.
        vector_dtype (str): The data type of the vector.
        pairwise_secrets (Dict[str, bytes]): A dictionary of pairwise secrets.
        pairwise_seeds (Dict[str, bytes]): A dictionary of pairwise seeds.
        my_node_id (str): The node ID of the current node.
    """
    def __init__(self, nonce: bytes) -> None:
        self.prf = PRF(nonce)
        self.vector_dtype = 'uint32'

    def setup_pairwise_secrets(self, my_node_id: str, nodes_ids: List[str]) -> None:
        """
        Sets up pairwise secrets for communication with other nodes.

        Args:
            my_node_id (str): The ID of the current node.
            nodes_ids (List[str]): A list of IDs of other nodes.
        """
        self.pairwise_secrets = {}
        self.pairwise_seeds = {}
        self.my_node_id = str(my_node_id)
        # this is just a hardcoded example
        for node_id in nodes_ids:
            if node_id != my_node_id:
                # create 32 bytes secret of zeros
                self.pairwise_secrets[node_id] = b'\x02' * 32

    def protect(self, tau: int, x_u_tau: List[int], node_ids: List[str]) -> List[int]:
        """
        Protects the input vector by applying a mask based on pairwise secrets.

        Args:
            tau (int): The current round number.
            x_u_tau (List[int]): The input vector to be protected.
            node_ids (List[str]): A list of node IDs to communicate with.

        Returns:
            List[int]: The protected (masked) vector.
        """
        x_u_tau = np.array(x_u_tau, dtype=self.vector_dtype)
        mask = np.zeros(len(x_u_tau), dtype=self.vector_dtype)
        for node_id in node_ids:
            if node_id == self.my_node_id:
                continue
            secret = self.pairwise_secrets[node_id]
            # generate seed for pairwise encryption
            pairwise_seed = self.prf.eval_key(pairwise_secret=secret, tau=tau)
            # expand seed to a random vector
            pairwise_vector = self.prf.eval_vector(seed=pairwise_seed, tau=tau, input_size=len(x_u_tau))
            pairwise_vector = np.frombuffer(pairwise_vector, dtype=self.vector_dtype)
            if node_id < self.my_node_id:
                mask += pairwise_vector
            else:
                mask -= pairwise_vector

        encrypted_params = mask + x_u_tau
        return encrypted_params.tolist()

    def aggregate(self, list_y_u_tau: Dict[str, np.ndarray]) -> List[int]:
        """
        Aggregates multiple vectors into a single vector.

        Args:
            list_y_u_tau (Dict[str, np.ndarray]): A dictionary of vectors from different nodes.

        Returns:
            List[int]: The aggregated vector.
        """
        list_y_u_tau = np.array(list_y_u_tau, dtype=self.vector_dtype)
        decrypted_vector = np.sum(list_y_u_tau, axis=0)
        print(decrypted_vector)
        decrypted_vector = decrypted_vector.astype(np.int32)
        decrypted_vector = decrypted_vector.tolist()

        return decrypted_vector
