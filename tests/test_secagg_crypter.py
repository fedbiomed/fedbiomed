import unittest

from math import ceil, log2
from unittest.mock import patch

from gmpy2 import mpz

from fedbiomed.common.secagg import SecaggCrypter, EncryptedNumber
from fedbiomed.common.secagg._jls import PublicParam
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError


class TestSecaggCrypter(unittest.TestCase):

    biprime = 158820908809271716671659880613366104677813341255487834154303909761107215283569995523817428402987962641429395032343305343341950966867458277812575065022203120547706127493272939455658018882112230042773163870472621818892994896895819790062496734944602899772583591514631486212290112369502692304700112819186167541107

    def setUp(self) -> None:
        self.secagg_crypter = SecaggCrypter()
        pass

    def tearDown(self) -> None:
        pass

    def tests_secagg_crypter_01_setup_public_param(self):
        """Tests biprime setup"""
        pp = self.secagg_crypter._setup_public_param(biprime=12345)
        self.assertIsInstance(pp, PublicParam)
        self.assertEqual(pp.n_modulus, 12345)

    def tests_secagg_crypter_02_convert_to_encrypted_number(self):
        """Tests convertion from int to encrypted number"""
        vector = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

        pp = self.secagg_crypter._setup_public_param(biprime=12345)

        l_encrypted_num = self.secagg_crypter._convert_to_encrypted_number(vector, pp)

        # Check each element of list is EncryptedNumber
        self.assertIsInstance(l_encrypted_num[0][0], EncryptedNumber)
        self.assertIsInstance(l_encrypted_num[2][3], EncryptedNumber)

        # Check if ciphertexts are in type of MPZ
        self.assertIsInstance(l_encrypted_num[2][3].ciphertext, mpz)
        self.assertIsInstance(l_encrypted_num[2][3].ciphertext, mpz)

        # Check ciphertext has correct value
        self.assertEqual(l_encrypted_num[0][0].ciphertext, 1)
        self.assertEqual(l_encrypted_num[2][3].ciphertext, 4)

    def test_secagg_crypter_03_apply_average(self):
        """Tests quantized divide"""

        vector = [4, 8, 12]
        result = self.secagg_crypter._apply_average(vector, 2, 0)

        # Test division
        self.assertListEqual(result, [v / 2 for v in vector])

    def test_secagg_crypter_03_encrypt(self):
        """Tests encryption"""

        key = 10
        params = [10.0, 20.0, 30.0, 40.0]
        current_round = 1
        num_nodes = 2

        result = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                             current_round=current_round,
                                             params=params,
                                             biprime=TestSecaggCrypter.biprime,
                                             key=key)

        self.assertIsInstance(result, list)

        # IMPORTANT = Bit size can change based on current_round and num_nodes
        self.assertEqual(ceil(log2(int(result[0]))), TestSecaggCrypter.biprime.bit_length() * 2)

        with self.assertRaises(FedbiomedSecaggCrypterError):
            result = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                                 current_round=current_round,
                                                 params="not-a-list",
                                                 biprime=TestSecaggCrypter.biprime,
                                                 key=key)

        with self.assertRaises(FedbiomedSecaggCrypterError):
            result = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                                 current_round=current_round,
                                                 params=["not-a-float", "not-a-float"],
                                                 biprime=TestSecaggCrypter.biprime,
                                                 key=key)
        # Not a float
        with self.assertRaises(FedbiomedSecaggCrypterError):
            result = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                                 current_round=current_round,
                                                 params=[0, 1, 2],
                                                 biprime=TestSecaggCrypter.biprime,
                                                 key=key)

        # If JLS.aggregate raises
        with patch("fedbiomed.common.secagg._jls.UserKey") as mock_user_key:
            # Invalid type of UserKey
            with self.assertRaises(FedbiomedSecaggCrypterError):
                result = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                                     current_round=current_round,
                                                     params=params,
                                                     biprime=TestSecaggCrypter.biprime,
                                                     key=key)

    def test_secagg_crypter_03_decrypt(self):
        """Tests decryption"""

        params = [0.5, 0.8, -0.5, 0.0]
        current_round = 2
        num_nodes = 2

        node_1 = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                             current_round=current_round,
                                             params=params,
                                             biprime=TestSecaggCrypter.biprime,
                                             key=10)

        node_2 = self.secagg_crypter.encrypt(num_nodes=num_nodes,
                                             current_round=current_round,
                                             params=params,
                                             biprime=TestSecaggCrypter.biprime,
                                             key=10)

        result = self.secagg_crypter.aggregate(current_round=current_round,
                                               num_nodes=num_nodes,
                                               params=[node_1, node_2],
                                               biprime=TestSecaggCrypter.biprime,
                                               key=-20,
                                               total_sample_size=8)

        print(result)
        self.assertEqual(len(result), len(params))
        self.assertTrue(result[0] > 0.4 or result[0] < 0.6,
                        "Secure aggregation result is not closer to expected avereage")

        # Test failure of JLS aggregate
        # If num of nodes does not match the number of parameters provided
        with patch("fedbiomed.common.secagg._secagg_crypter.SecaggCrypter._convert_to_encrypted_number") as m:
            def side_effect(x, p):
                return x
            m.side_effect = side_effect
            with self.assertRaises(FedbiomedSecaggCrypterError):
                self.secagg_crypter.aggregate(current_round=current_round,
                                              num_nodes=num_nodes,
                                              params=[node_1, node_2],
                                              biprime=TestSecaggCrypter.biprime,
                                              key=-20,
                                              total_sample_size=8)

        # Raise if num nodes not equal to number fo parameters
        with self.assertRaises(FedbiomedSecaggCrypterError):
            params = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
            self.secagg_crypter.aggregate(current_round=current_round,
                                          num_nodes=num_nodes,
                                          params=params,
                                          biprime=TestSecaggCrypter.biprime,
                                          key=-20,
                                          total_sample_size=8)

        # Invalid type of single parameter
        with self.assertRaises(FedbiomedSecaggCrypterError):
            params = [["not-int", "not-int", "not-int"], ["not-int", "not-int", "not-int"]]
            self.secagg_crypter.aggregate(current_round=current_round,
                                          num_nodes=num_nodes,
                                          params=params,
                                          key=-20,
                                          biprime=TestSecaggCrypter.biprime,
                                          total_sample_size=8)

        # Unsupported list shape
        with self.assertRaises(FedbiomedSecaggCrypterError):
            params = ["not-int", "not-int"]
            self.secagg_crypter.aggregate(current_round=current_round,
                                          num_nodes=num_nodes,
                                          params=params,
                                          key=-20,
                                          biprime=TestSecaggCrypter.biprime,
                                          total_sample_size=8)



if __name__ == "__main__":
    unittest.main()
