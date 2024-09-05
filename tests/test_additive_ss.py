import unittest
from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.secagg._additive_ss import Secret, Share

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
        for share in shares:
            self.assertTrue(-2**10 <= share.value <=2**10)

    def test_reconstruct_secret_int(self):
        """Test reconstructing an integer secret."""
        shares = [Share(10), Share(20), Share(15)]
        reconstructed = Secret.reconstruct(shares)
        self.assertEqual(reconstructed, 45)

    def test_reconstruct_secret_list(self):
        """Test reconstructing a list secret."""
        shares = [Share([1, 2, 3]), Share([4, 5, 6]), Share([5, 13, 21])] 
        reconstructed = Secret.reconstruct(shares)
        self.assertEqual(reconstructed, [10, 20, 30])

    def test_reconstruct_invalid_shares(self):
        """Test reconstructing with invalid shares."""
        invalid_shares = [Share(10), Share([20, 30])]
        with self.assertRaises(FedbiomedTypeError):
            Secret.reconstruct(invalid_shares)


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
        share1 = Secret(10).split(2)
        share2 = Secret(15).split(2)
        result = Share.add(share1, share2)
        reconstructed = Secret.reconstruct(result)
        self.assertEqual(reconstructed, 25)

    def test_add_shares_list(self):
        """Test adding two list shares."""
        share1 = Secret([1, 2, 3]).split(3)
        share2 = Secret([4, 5, 6]).split(3)
        result = Share.add(share1, share2)
        reconstructed = Secret.reconstruct(result)
        self.assertEqual(reconstructed, [5, 7, 9])

    def test_add_invalid_shares(self):
        """Test adding shares of different types (int and list)."""
        share1 = Secret(10).split(2)
        share2 = Secret([1, 2, 3]).split(2)
        with self.assertRaises(FedbiomedTypeError):
            Share.add(share1, share2)

    def test_add_share_and_int(self):
        """Test adding a Share object and an integer."""
        share = Share(10)
        result = share + 5
        self.assertEqual(result.value, 15)

    def test_add_share_and_list(self):
        """Test adding a Share object and a list."""
        share = Share([1, 2, 3])
        result = share + [4, 5, 6]
        self.assertEqual(result.value, [5, 7, 9])


if __name__ == '__main__':
    unittest.main()
