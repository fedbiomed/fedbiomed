import unittest
from fedbiomed.common.secagg import Secret, Share
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers

# Assuming the classes Secret and Share are defined in secret_module
# from secret_module import Secret, Share


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
        with self.assertRaises(FedbiomedSecaggCrypterError) as context:
            Secret("invalid_secret")
        
    def test_reconstruct_secret_int(self):
        """Test reconstructing an integer secret."""
        shares = [Share(10), Share(20), Share(15)]
        reconstructed = Secret.reconstruct(shares)
        self.assertEqual(reconstructed, 45)

    def test_reconstruct_secret_list(self):
        """Test reconstructing a list secret."""
        shares = [Share([1, 2, 3]), Share([4, 5, 6]), Share([5, 13, 21])]
        secret = [10, 20, 30]
        reconstructed = Secret.reconstruct(shares)
        self.assertEqual(reconstructed, [10, 20, 30])

    def test_reconstruct_invalid_shares(self):
        """Test reconstructing with invalid shares."""
        with self.assertRaises(FedbiomedSecaggCrypterError):
            Secret.reconstruct([Share(10), Share([20, 30])])

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
        with self.assertRaises(FedbiomedSecaggCrypterError):
            Share("invalid_share")

    def test_add_shares_int(self):
        """Test adding two integer shares."""
        share1 = Share(10)
        share2 = Share(15)
        result = share1 + share2
        self.assertEqual(result.value, 25)

    def test_add_shares_list(self):
        """Test adding two list shares."""
        share1 = Share([1, 2, 3])
        share2 = Share([4, 5, 6])
        result = share1 + share2
        self.assertEqual(result.value, [5, 7, 9])

    def test_add_invalid_shares(self):
        """Test adding shares of different types (int and list)."""
        share1 = Share(10)
        share2 = Share([1, 2, 3])
        with self.assertRaises(FedbiomedSecaggCrypterError):
            share1 + share2



if __name__ == '__main__':
    unittest.main()
