import unittest
from cryptography.hazmat.primitives.asymmetric import ec
from fedbiomed.common.secagg._dh import DHKey, DHKeyAgreement
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers


class TestDHKey(unittest.TestCase):

    def setUp(self):
        """Initialize the DHKey instance."""
        self.dh_keys = DHKey()

    def test_dhkey_01_private_key_export(self):
        """Test if the exported private key is a bytes object."""
        private_key_pem = self.dh_keys.export_private_key()
        self.assertTrue(isinstance(private_key_pem, bytes))

    def test_dhkey_02_public_key_export(self):
        """Test if the exported public key is a bytes object."""
        public_key_pem = self.dh_keys.export_public_key()
        self.assertTrue(isinstance(public_key_pem, bytes))

    def test_dhkey_03_import_private_key(self):
        """Test importing a private key."""
        private_key_pem = self.dh_keys.export_private_key()
        dh_key_imported = DHKey(private_key_pem=private_key_pem)
        self.assertTrue(
            isinstance(dh_key_imported.private_key, ec.EllipticCurvePrivateKey)
        )

    def test_dhkey_04_import_public_key(self):
        """Test importing a public key."""
        public_key_pem = self.dh_keys.export_public_key()
        dh_key_imported = DHKey(public_key_pem=public_key_pem)
        self.assertTrue(
            isinstance(dh_key_imported.public_key, ec.EllipticCurvePublicKey)
        )

    def test_dhkey_05_import_invalid_private_key(self):
        """Test importing an invalid private key raises an exception."""
        with self.assertRaises(FedbiomedSecaggCrypterError):
            DHKey(private_key_pem=b"invalid_key_data")

    def test_dhkey_06_import_invalid_public_key(self):
        """Test importing an invalid public key raises an exception."""
        with self.assertRaises(FedbiomedSecaggCrypterError):
            DHKey(public_key_pem=b"invalid_key_data")


class TestDHKeyAgreement(unittest.TestCase):

    def setUp(self):
        """Initialize the DHKeyAgreement instances."""
        self.node_u_id = "node_u"
        self.node_v_id = "node_v"
        session_salt = b"this_is_a_salt"

        self.dh_keys_u = DHKey()
        self.dh_keys_v = DHKey()

        self.node_u_private_key_pem = self.dh_keys_u.export_private_key()
        self.node_v_private_key_pem = self.dh_keys_v.export_private_key()
        self.node_u_public_key_pem = self.dh_keys_u.export_public_key()
        self.node_v_public_key_pem = self.dh_keys_v.export_public_key()

        self.dh_agreement_u = DHKeyAgreement(
            self.node_u_id, DHKey(self.node_u_private_key_pem), session_salt
        )
        self.dh_agreement_v = DHKeyAgreement(
            self.node_v_id, DHKey(self.node_v_private_key_pem), session_salt
        )

    def test_dhkey_agreement_01_kdf(self):
        """Test if the derived key is a bytes object."""
        pairwise_key = self.dh_agreement_u._kdf(b"secret_key", self.node_v_id)
        self.assertTrue(isinstance(pairwise_key, bytes))
        self.assertEqual(len(pairwise_key), 32)

    def test_dhkey_agreement_02_agree(self):
        """Test if the agreed key is a bytes object."""
        agreed_key_u = self.dh_agreement_u.agree(
            self.node_v_id, self.node_v_public_key_pem
        )
        self.assertTrue(isinstance(agreed_key_u, bytes))

    def test_dhkey_agreement_03_agree_invalid_key_type(self):
        """Test if agreeing with an invalid key type raises an exception."""
        with self.assertRaises(FedbiomedSecaggCrypterError):
            self.dh_agreement_u.agree(self.node_v_id, self.node_u_private_key_pem)

    def test_dhkey_agreement_04_key_agreement_consistency(self):
        """Test if the agreed keys are the same for both nodes."""
        agreed_key_u_to_v = self.dh_agreement_u.agree(
            self.node_v_id, self.node_v_public_key_pem
        )
        agreed_key_v_to_u = self.dh_agreement_v.agree(
            self.node_u_id, self.node_u_public_key_pem
        )
        # the shared secret should be the same for both nodes, s_u_v = s_v_u
        self.assertEqual(agreed_key_u_to_v, agreed_key_v_to_u)


if __name__ == "__main__":
    unittest.main()
