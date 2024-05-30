import unittest
from Crypto.PublicKey import ECC
from fedbiomed.common.secagg._dh import DHKeysGeneration, DHKeyAgreement
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers

class TestDHKeysGeneration(unittest.TestCase):

    def setUp(self):
        """ Initialize the DHKeysGeneration instance."""
        self.dh_keys = DHKeysGeneration()

    def test_private_key_export(self):
        """ Test if the exported private key is a bytes object."""
        private_key_pem = self.dh_keys.export_private_key()
        self.assertTrue(isinstance(private_key_pem, bytes))

    def test_public_key_export(self):
        """ Test if the exported public key is a bytes object."""
        public_key_pem = self.dh_keys.export_public_key()
        self.assertTrue(isinstance(public_key_pem, bytes))


class TestDHKeyAgreement(unittest.TestCase):

    def setUp(self):
        """ Initialize the DHKeyAgreement instances."""
        self.node_u_id = 'node_u'
        self.node_v_id = 'node_v'
        session_salt = b'this_is_a_salt'

        self.dh_keys_u = DHKeysGeneration()
        self.dh_keys_v = DHKeysGeneration()

        self.node_u_private_key_pem = self.dh_keys_u.export_private_key()
        self.node_v_private_key_pem = self.dh_keys_v.export_private_key()
        self.node_u_public_key_pem = self.dh_keys_u.export_public_key()
        self.node_v_public_key_pem = self.dh_keys_v.export_public_key()

        self.dh_agreement_u = DHKeyAgreement(self.node_u_id, self.node_u_private_key_pem, session_salt)
        self.dh_agreement_v = DHKeyAgreement(self.node_v_id, self.node_v_private_key_pem, session_salt)

    def test_import_key_pem(self):
        """ Test if the imported key is an ECC.EccKey object. """
        imported_key = DHKeyAgreement._import_key_pem(self.node_u_private_key_pem, "PRIVATE")
        self.assertTrue(isinstance(imported_key, ECC.EccKey))

    def test_import_key_pem_invalid_type(self):
        """ Test if importing an invalid key type raises an exception. """
        with self.assertRaises(FedbiomedSecaggCrypterError):
            DHKeyAgreement._import_key_pem(self.node_u_private_key_pem, "INVALID_TYPE")

    def test_kdf(self):
        """ Test if the derived key is a bytes object. """
        pairwise_key = self.dh_agreement_u._kdf(b'secret_key', self.node_v_id)
        self.assertTrue(isinstance(pairwise_key, bytes))
        self.assertEqual(len(pairwise_key), 32)

    def test_agree(self):
        """ Test if the agreed key is a bytes object. """
        agreed_key_u = self.dh_agreement_u.agree(self.node_v_id, self.node_v_public_key_pem)
        self.assertTrue(isinstance(agreed_key_u, bytes))
    
    def test_agree_invalid_key_type(self):
        """ Test if agreeing with an invalid key type raises an exception. """
        with self.assertRaises(FedbiomedSecaggCrypterError):
            self.dh_agreement_u.agree(self.node_v_id, self.node_u_private_key_pem)

    def test_key_agreement_consistency(self):
        """ Test if the agreed keys are the same for both nodes. """
        agreed_key_u_to_v = self.dh_agreement_u.agree(self.node_v_id, self.node_v_public_key_pem)
        agreed_key_v_to_u = self.dh_agreement_v.agree(self.node_u_id, self.node_u_public_key_pem)
        # the shared secret should be the same for both nodes, s_u_v = s_v_u
        self.assertEqual(agreed_key_u_to_v, agreed_key_v_to_u)

if __name__ == '__main__':
    unittest.main()
