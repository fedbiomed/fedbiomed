import unittest
import testsupport.mock_node_environ  # noqa (remove flake8 false warning)
import numpy as np

from unittest.mock import patch
from torch.utils.data import Dataset, Subset
from fedbiomed.common.data import TorchTabularDataset



class TestTorchDataManager(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_torch_data_manager_01_init_failure(self):
            pass

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
