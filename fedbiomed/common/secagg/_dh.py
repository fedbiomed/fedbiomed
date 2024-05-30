from Crypto.PublicKey import ECC
from Crypto.Hash import TupleHash128
from Crypto.Protocol.DH import key_agreement
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers


class DHKeysGeneration:
    """
    Key Generation with ephemeral-ephemeral ECDH, a session is defined for each experiment.

    This class handles the generation of ECC keys using the P-256 curve. It provides methods
    to export the private and public keys in PEM format as bytes.

    Attributes:
        private_key (ECC.EccKey): The user's private ECC key.
        public_key (ECC.EccKey): The user's public ECC key.
    """

    def __init__(self):
        """
        Initializes the DHKeysGeneration instance by generating a private key and deriving the public key.
        """
        self._private_key: ECC.EccKey = ECC.generate(curve="P-256")
        self._public_key: ECC.EccKey = self._private_key.public_key()

    def export_private_key(self) -> bytes:
        """
        Exports the private key to PEM format.

        Returns:
            bytes: The private key in PEM format.
        """
        return self._private_key.export_key(format="PEM").encode("utf-8")

    def export_public_key(self) -> bytes:
        """
        Exports the public key to PEM format.

        Returns:
            bytes: The public key in PEM format.
        """
        return self._public_key.export_key(format="PEM").encode("utf-8")


class DHKeyAgreement:
    """
    Key Agreement with ephemeral-ephemeral ECDH, a session is defined for each experiment.

    This class handles the key agreement process using the private ECC key of the user. It imports
    the private key, performs the key exchange, and derives the shared secret using a KDF.

    Attributes:
        private_key (ECC.EccKey): The user's private ECC key.
    """

    def __init__(self, node_u_id: str, node_u_private_key: bytes, session_salt: bytes):
        """
        Initializes the DHKeyAgreement instance.

        Args:
            node_u_id (str): The ID of the node.
            node_u_private_key (bytes): The private key in PEM format as bytes.
            session_salt (bytes): A session-specific salt.
        """
        self._node_u_id: str = node_u_id
        self._private_key: ECC.EccKey = self._import_key_pem(node_u_private_key, key_type="PRIVATE")
        self.session_salt = session_salt

    @staticmethod
    def _import_key_pem(key: bytes, key_type: str) -> ECC.EccKey:
        """
        Imports the ECC key from PEM format bytes.

        Args:
            key (bytes): The key in PEM format as bytes.
            key_type (str): The type of key ("PUBLIC" or "PRIVATE").

        Returns:
            ECC.EccKey: The imported ECC key.
        """
        key_pem = key.decode("utf-8")
        if key_type in key_pem:
            return ECC.import_key(key_pem)
        else:
            raise FedbiomedSecaggCrypterError(f"{ErrorNumbers.FB629.value}: Invalid key type, expected '{key_type}'")

    def _kdf(self, key: bytes, node_v_id: str) -> bytes:
        """
        Key Derivation Function (KDF) that derives a 32-byte key using TupleHash128.

        Args:
            key (bytes): The shared secret key.
            node_v_id (str): The ID of the other node (node_v).

        Returns:
            bytes: The derived key of 32 bytes.
        """
        h = TupleHash128.new(digest_bytes=32)
        # h.update is not commutative, the order of the arguments is important so we combine the node ids w.r.t the order
        node_ids = self._node_u_id + node_v_id if self._node_u_id < node_v_id else node_v_id + self._node_u_id
        h.update(key + self.session_salt + node_ids.encode("utf-8"))
        return h.digest()

    def agree(self, node_v_id: str, public_key_pem: bytes) -> bytes:
        """
        Performs the key agreement and derives the shared secret.

        Args:
            node_v_id (str): The ID of the other node (node_v).
            public_key_pem (bytes): The public key of the other node in PEM format as bytes.

        Returns:
            bytes: The derived shared secret.
        """
        public_key = self._import_key_pem(public_key_pem, key_type="PUBLIC")
        shared_secret = key_agreement(
            static_priv=self._private_key,
            static_pub=public_key,
            kdf=lambda key: self._kdf(key, node_v_id)
        )
        return shared_secret
