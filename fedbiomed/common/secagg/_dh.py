from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash
from cryptography.hazmat.backends import default_backend

from typing import Callable

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers


class DHKey:
    """
    Key handling for ephemeral-ephemeral ECDH, a session is defined for each experiment.

    This class handles the generation and importing of ECC keys using the P-256 curve. It provides methods
    to export the private and public keys in PEM format as bytes.

    Attributes:
        private_key: The user's private ECC key.
        public_key: The user's public ECC key.
    """

    def __init__(
        self,
        private_key_pem: bytes | None = None,
        public_key_pem: bytes | None = None
    ) -> None:
        """
        Initializes the DHKey instance by generating a new key pair or importing the provided keys.

        Args:
            private_key_pem: Optional. The private key in PEM format.
            public_key_pem: Optional. The public key in PEM format.
        """
        if private_key_pem:
            self.private_key = self._import_key(
                serialization.load_pem_private_key,
                data=private_key_pem,
                password=None,
                backend=default_backend()
            )
        elif not public_key_pem:
            self.private_key = ec.generate_private_key(
                ec.SECP256R1(), default_backend()
            )
        else:
            # Means that only public key is loaded
            self.private_key = None

        if public_key_pem:
            self.public_key = self._import_key(serialization.load_pem_public_key,
                                               data=public_key_pem,
                                               backend=default_backend()
                                               )
        else:
            self.public_key = self.private_key.public_key()


    def export_private_key(self):
        """
        Exports the private key to PEM format.

        Returns:
            The private key in PEM format. Returns `None` if private key
                is not loaded.
        """
        if not self.private_key:
            return None

        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def export_public_key(self):
        """
        Exports the public key to PEM format.

        Returns:
            The public key in PEM format.
        """
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    @staticmethod
    def _import_key(func: Callable, **kwargs):
        """Executes import function

        Args:
            func: private key import function or public key import function
            kwargs: Argument to pass to the `func`
        """
        try:
            return func(**kwargs)
        except ValueError as exp:
            raise FedbiomedSecaggCrypterError(
                f"{ErrorNumbers.FB629.value}: Invalid key format, {kwargs}"
            ) from exp



class DHKeyAgreement:
    """
    Key Agreement with ephemeral-ephemeral ECDH, a session is defined for each experiment.

    This class handles the key agreement process using the private ECC key of the user. It imports
    the private key, performs the key exchange, and derives the shared secret using a KDF.

    Attributes:
        private_key: The user's private ECC key.
    """

    def __init__(self, node_u_id, node_u_dh_key: DHKey, session_salt):
        """
        Initializes the DHKeyAgreement instance.

        Args:
            node_u_id: The ID of the node.
            node_u_dh_key: The keypair of the node.
            session_salt: A session-specific salt.
        """
        self._node_u_id = node_u_id
        self._dh_key = node_u_dh_key
        self.session_salt = session_salt

    def _kdf(self, key, node_v_id):
        """
        Key Derivation Function (KDF) that derives a 32-byte key using ConcatKDFHash.

        Args:
            key: The shared secret key.
            node_v_id: The ID of the other node (node_v).

        Returns:
            The derived key of 32 bytes.
        """
        node_ids = (
            self._node_u_id + node_v_id
            if self._node_u_id < node_v_id
            else node_v_id + self._node_u_id
        )
        ckdf = ConcatKDFHash(
            algorithm=hashes.SHA256(),
            length=32,
            otherinfo=self.session_salt + node_ids.encode("utf-8"),
            backend=default_backend(),
        )
        return ckdf.derive(key)

    def agree(self, node_v_id, public_key_pem):
        """
        Performs the key agreement and derives the shared secret.

        Args:
            node_v_id: The ID of the other node (node_v).
            public_key_pem: The public key of the other node in PEM format as bytes.

        Returns:
            The derived shared secret.
        """
        dh_v_key = DHKey(public_key_pem=public_key_pem)
        shared_secret = self._dh_key.private_key.exchange(ec.ECDH(), dh_v_key.public_key)
        derived_key = self._kdf(shared_secret, node_v_id)
        return derived_key
