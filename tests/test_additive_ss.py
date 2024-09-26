import unittest
from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.secagg._additive_ss import AdditiveSecret, AdditiveShare, AdditiveShares
import random

class TestAdditiveSecret(unittest.TestCase):

    def test_secret_initialization_valid_int(self):
        """Test initializing AdditiveSecret with a valid integer."""
        secret = AdditiveSecret(123)
        self.assertEqual(secret.secret, 123)

    def test_secret_initialization_valid_list(self):
        """Test initializing AdditiveSecret with a valid list of integers."""
        secret = AdditiveSecret([1, 2, 3])
        self.assertEqual(secret.secret, [1, 2, 3])

    def test_secret_initialization_invalid_type(self):
        """Test initializing AdditiveSecret with an invalid type."""
        with self.assertRaises(FedbiomedValueError) as context:
            AdditiveSecret("invalid_secret")
        self.assertIn("AdditiveSecret must be an int or a list of int", str(context.exception))

    def test_split_invalid_num_shares(self):
        """Test splitting with invalid number of shares."""
        secret = AdditiveSecret(10)
        with self.assertRaises(FedbiomedValueError):
            secret.split(0)
        with self.assertRaises(FedbiomedValueError):
            secret.split(-5)

    def test_split_with_bit_length(self):
        """Test splitting with a specific bit length."""
        secret = AdditiveSecret(1024)
        shares = secret.split(3, bit_length=10)

        # Ensure all shares are within the specified bit length range
        for share in shares:
            self.assertTrue(-2**10 <= share.value <=2**10)

    def test_reconstruct_secret_int(self):
        """Test reconstructing an integer secret."""
        shares = AdditiveShares([AdditiveShare(10), AdditiveShare(20), AdditiveShare(15)])
        reconstructed = shares.reconstruct()
        self.assertEqual(reconstructed, 45)

    def test_reconstruct_secret_list(self):
        """Test reconstructing a list secret."""
        shares = AdditiveShares([AdditiveShare([1, 2, 3]), AdditiveShare([4, 5, 6]), AdditiveShare([5, 13, 21])])
        reconstructed = shares.reconstruct()
        self.assertEqual(reconstructed, [10, 20, 30])

    def test_reconstruct_invalid_shares(self):
        """Test reconstructing with invalid shares."""
        invalid_shares = AdditiveShares([AdditiveShare(10), AdditiveShare([20, 30])])
        with self.assertRaises(FedbiomedTypeError):
            invalid_shares.reconstruct()


class TestAdditiveShare(unittest.TestCase):

    def test_share_initialization_valid_int(self):
        """Test initializing AdditiveShare with a valid integer."""
        share = AdditiveShare(123)
        self.assertEqual(share.value, 123)

    def test_share_initialization_valid_list(self):
        """Test initializing AdditiveShare with a valid list of integers."""
        share = AdditiveShare([1, 2, 3])
        self.assertEqual(share.value, [1, 2, 3])

    def test_share_initialization_invalid_type(self):
        """Test initializing AdditiveShare with an invalid type."""
        with self.assertRaises(FedbiomedTypeError):
            AdditiveShare("invalid_share")

    def test_add_shares_int(self):
        """Test adding two integer shares."""
        shares_1 = AdditiveSecret(10).split(2)
        shares_2 = AdditiveSecret(15).split(2)
        result = shares_1 + shares_2
        reconstructed = result.reconstruct()
        self.assertEqual(reconstructed, 25)

    def test_add_shares_list(self):
        """Test adding two list shares."""
        shares_1 = AdditiveSecret([1, 2, 3]).split(3)
        shares_2 = AdditiveSecret([4, 5, 6]).split(3)
        result = shares_1 + shares_2
        reconstructed = result.reconstruct()
        self.assertEqual(reconstructed, [5, 7, 9])

    def test_add_invalid_shares(self):
        """Test adding shares of different types (int and list)."""
        shares_1 = AdditiveSecret(10).split(2)
        shares_2 = AdditiveSecret([1, 2, 3]).split(2)
        with self.assertRaises(FedbiomedTypeError):
            shares_1 + shares_2

    def test_ahe_setup_int(self):
        """Test to reproduce the setup for additive homomorphic encryption scheme (Joye Libert)."""
        # generate 3 users' keys
        user_key_1 = random.randint(0, 2**2048)
        user_key_2 = random.randint(0, 2**2048)
        user_key_3 = random.randint(0, 2**2048)
        # generate 3 users' shares
        shares_1 = AdditiveSecret(user_key_1).split(3)
        shares_2 = AdditiveSecret(user_key_2).split(3)
        shares_3 = AdditiveSecret(user_key_3).split(3)
        # each user send n-1 shares to the other users
        user_1_to_2 = shares_1[1]
        user_1_to_3 = shares_1[2]

        user_2_to_1 = shares_2[0]
        user_2_to_3 = shares_2[2]

        user_3_to_1 = shares_3[0]
        user_3_to_2 = shares_3[1]
        # each user reconstruct the secret
        server_key_shares_1 = shares_1[0] + user_2_to_1 + user_3_to_1
        server_key_shares_2 = shares_2[1] + user_1_to_2 + user_3_to_2
        server_key_shares_3 = shares_3[2] + user_1_to_3 + user_2_to_3

        # reseacher reconstruct the server key
        server_key = AdditiveShares([server_key_shares_1,server_key_shares_2,server_key_shares_3]).reconstruct()

        original_key = user_key_1 + user_key_2 + user_key_3

        self.assertEqual(server_key, original_key)

    def test_ahe_setup_list(self):
        """Test to reproduce the setup for additive homomorphic encryption scheme (Learning With Error SA) with list shares."""
        # generate 3 users' keys
        user_key_1 = [random.randint(0, 2**50) for _ in range(10)]
        user_key_2 = [random.randint(0, 2**50) for _ in range(10)]
        user_key_3 = [random.randint(0, 2**50) for _ in range(10)]

        # generate 3 users' shares
        shares_1 = AdditiveSecret(user_key_1).split(3)
        shares_2 = AdditiveSecret(user_key_2).split(3)
        shares_3 = AdditiveSecret(user_key_3).split(3)

        # each user send n-1 shares to the other users
        user_1_to_2 = shares_1[1]
        user_1_to_3 = shares_1[2]

        user_2_to_1 = shares_2[0]
        user_2_to_3 = shares_2[2]

        user_3_to_1 = shares_3[0]
        user_3_to_2 = shares_3[1]

        # each user reconstruct the secret
        server_key_shares_1 = shares_1[0] + user_2_to_1 + user_3_to_1

        server_key_shares_2 = shares_2[1] + user_1_to_2 + user_3_to_2
        server_key_shares_3 = shares_3[2] + user_1_to_3 + user_2_to_3

        # reseacher reconstruct the server key
        server_key = AdditiveShares([server_key_shares_1,server_key_shares_2,server_key_shares_3]).reconstruct()

        original_key = [user_key_1[i] + user_key_2[i] + user_key_3[i] for i in range(10)]

        self.assertEqual(server_key, original_key)



if __name__ == '__main__':
    unittest.main()
