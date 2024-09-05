import unittest
import random
from fedbiomed.common.secagg._additive_ss import Secret, Share 

class TestSecret(unittest.TestCase):

    def setUp(self):
        """Initialize with example secrets."""
        # Initialize with simple test cases
        self.secret_int = 100
        self.secret_list = [10, 20, 30]
        random.seed(42)  # Seed for reproducibility of randomness

    def test_secret_01_int_split(self):
        """Test splitting an integer secret into shares."""
        num_shares = 3
        secret_obj = Secret(self.secret_int)
        shares = secret_obj.split(num_shares)

        # Check if the correct number of shares is generated
        self.assertEqual(len(shares), num_shares)

        # Check that the sum of all shares reconstructs the original secret
        sum_of_shares = sum(share.value for share in shares)
        self.assertEqual(sum_of_shares, self.secret_int)

    def test_secret_02_list_split(self):
        """Test splitting a list of integers secret into shares."""
        num_shares = 3
        secret_obj = Secret(self.secret_list)
        shares = secret_obj.split(num_shares)

        # Check if the correct number of shares is generated
        self.assertEqual(len(shares), num_shares)

        # Ensure that each share is a list of the same length as the original secret list
        for share in shares:
            self.assertTrue(isinstance(share.value, list))
            self.assertEqual(len(share.value), len(self.secret_list))

        # Check that the sum of shares reconstructs the original list
        sum_of_shares = [sum(x) for x in zip(*(share.value for share in shares))]
        self.assertEqual(sum_of_shares, self.secret_list)

    def test_secret_03_invalid_secret_type(self):
        """Test initializing a Secret with an invalid type."""
        with self.assertRaises(ValueError):
            Secret("invalid_type")  # Should raise an error

    def test_secret_04_invalid_split_type(self):
        """Test invalid split type during share generation."""
        with self.assertRaises(ValueError):
            Secret({1, 2, 3}).split(3)  # Pass an invalid secret type


class TestShare(unittest.TestCase):

    def setUp(self):
        """Initialize with some example shares."""
        random.seed(42)  # Seed for reproducibility of randomness
        self.share_int_1 = Share(50)
        self.share_int_2 = Share(50)
        self.share_list_1 = Share([1, 2, 3])
        self.share_list_2 = Share([4, 5, 6])

    def test_share_01_add_int(self):
        """Test adding two integer shares."""
        result_share = self.share_int_1 + self.share_int_2
        self.assertTrue(isinstance(result_share, Share))
        self.assertEqual(result_share.value, 100)

    def test_share_02_add_list(self):
        """Test adding two list shares."""
        result_share = self.share_list_1 + self.share_list_2
        self.assertTrue(isinstance(result_share, Share))
        self.assertEqual(result_share.value, [5, 7, 9])

    def test_share_03_add_mismatched_type(self):
        """Test adding mismatched types raises an error."""
        with self.assertRaises(TypeError):
            self.share_int_1 + self.share_list_1  # Should raise TypeError due to type mismatch

    def test_share_04_repr(self):
        """Test the __repr__ method for Share class."""
        self.assertEqual(repr(self.share_int_1), "Share(50)")
        self.assertEqual(repr(self.share_list_1), "Share([1, 2, 3])")


if __name__ == "__main__":
    unittest.main()
