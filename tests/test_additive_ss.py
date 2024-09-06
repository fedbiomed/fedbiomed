import unittest
from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.secagg._additive_ss import Secret, Share, Shares
import random

class TestSecret(unittest.TestCase):
    
    def test_secret_initialization_valid_int(self):
        """Test initializing Secret with a valid integer."""
        secret = Secret(123)
        self.assertEqual(secret.secret, 123)

    def test_secret_initialization_valid_list(self):
        """Test initializing Secret with a valid list of integers."""
        secret = Secret([1, 2, 3])
        self.assertEqual(secret.secret, [1, 2, 3])

    def test_secret_initialization_invalid_type(self):
        """Test initializing Secret with an invalid type."""
        with self.assertRaises(FedbiomedValueError) as context:
            Secret("invalid_secret")
        self.assertIn("Secret must be an int or a list of int", str(context.exception))
    
    def test_split_invalid_num_shares(self):
        """Test splitting with invalid number of shares."""
        secret = Secret(10)
        with self.assertRaises(FedbiomedValueError):
            secret.split(0)
        with self.assertRaises(FedbiomedValueError):
            secret.split(-5)

    def test_split_with_bit_length(self):
        """Test splitting with a specific bit length."""
        secret = Secret(1024)
        shares = secret.split(3, bit_length=10)
        
        # Ensure all shares are within the specified bit length range
        for share in shares.values:
            self.assertTrue(-2**10 <= share <=2**10)

    def test_reconstruct_secret_int(self):
        """Test reconstructing an integer secret."""
        shares = Shares([Share(10), Share(20), Share(15)])
        reconstructed = shares.reconstruct()
        self.assertEqual(reconstructed, 45)

    def test_reconstruct_secret_list(self):
        """Test reconstructing a list secret."""
        shares = Shares([Share([1, 2, 3]), Share([4, 5, 6]), Share([5, 13, 21])])
        reconstructed = shares.reconstruct()
        self.assertEqual(reconstructed, [10, 20, 30])

    def test_reconstruct_invalid_shares(self):
        """Test reconstructing with invalid shares."""
        invalid_shares = Shares([Share(10), Share([20, 30])])
        with self.assertRaises(FedbiomedTypeError):
            invalid_shares.reconstruct()


class TestShare(unittest.TestCase):

    def test_share_initialization_valid_int(self):
        """Test initializing Share with a valid integer."""
        share = Share(123)
        self.assertEqual(share.value, 123)

    def test_share_initialization_valid_list(self):
        """Test initializing Share with a valid list of integers."""
        share = Share([1, 2, 3])
        self.assertEqual(share.value, [1, 2, 3])

    def test_share_initialization_invalid_type(self):
        """Test initializing Share with an invalid type."""
        with self.assertRaises(FedbiomedTypeError):
            Share("invalid_share")

    def test_add_shares_int(self):
        """Test adding two integer shares."""
        shares_1 = Secret(10).split(2)
        shares_2 = Secret(15).split(2)
        result = shares_1 + shares_2
        reconstructed = result.reconstruct()
        self.assertEqual(reconstructed, 25)

    def test_add_shares_list(self):
        """Test adding two list shares."""
        shares_1 = Secret([1, 2, 3]).split(3)
        shares_2 = Secret([4, 5, 6]).split(3)
        result = shares_1 + shares_2
        reconstructed = result.reconstruct()
        self.assertEqual(reconstructed, [5, 7, 9])

    def test_add_invalid_shares(self):
        """Test adding shares of different types (int and list)."""
        shares_1 = Secret(10).split(2)
        shares_2 = Secret([1, 2, 3]).split(2)
        with self.assertRaises(FedbiomedTypeError):
            shares_1 + shares_2

    def test_ahe_setup_int(self):
        """Test to reproduce the setup for additive homomorphic encryption scheme (Joye Libert)."""
        # generate 3 users' keys
        user_key_1 = random.randint(0, 2**2048)
        user_key_2 = random.randint(0, 2**2048)
        user_key_3 = random.randint(0, 2**2048)
        # generate 3 users' shares
        shares_1 = Secret(user_key_1).split(3)
        shares_2 = Secret(user_key_2).split(3)
        shares_3 = Secret(user_key_3).split(3)
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
        server_key = Shares([server_key_shares_1,server_key_shares_2,server_key_shares_3]).reconstruct()

        original_key = user_key_1 + user_key_2 + user_key_3

        self.assertEqual(server_key, original_key)
        
    def test_ahe_setup_list(self):
        """Test to reproduce the setup for additive homomorphic encryption scheme (Learning With Error SA) with list shares."""
        # generate 3 users' keys
        user_key_1 = [random.randint(0, 2**50) for _ in range(10)]
        user_key_2 = [random.randint(0, 2**50) for _ in range(10)]
        user_key_3 = [random.randint(0, 2**50) for _ in range(10)]

        # generate 3 users' shares
        shares_1 = Secret(user_key_1).split(3)
        shares_2 = Secret(user_key_2).split(3)
        shares_3 = Secret(user_key_3).split(3)

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
        server_key = Shares([server_key_shares_1,server_key_shares_2,server_key_shares_3]).reconstruct()

        original_key = [user_key_1[i] + user_key_2[i] + user_key_3[i] for i in range(10)]

        self.assertEqual(server_key, original_key)



if __name__ == '__main__':
    unittest.main()
