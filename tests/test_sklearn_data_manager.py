import unittest
import testsupport.mock_node_environ  # noqa (remove flake8 false warning)
import numpy as np
import pandas as pd

from fedbiomed.common.data._sklearn_data_manager import SkLearnDataManager
from fedbiomed.common.exceptions import FedbiomedSkLearnDataManagerError


class TestSkLearnDataManager(unittest.TestCase):

    def setUp(self):
        # Setup global TorchDataset class
        self.inputs = np.array([[1, 2, 3, 4],
                                [1, 2, 3, 4],
                                [1, 2, 3, 4],
                                [1, 2, 3, 4]
                                ])
        self.target = np.array([1, 2, 3, 4])
        self.sklearn_dataset = SkLearnDataManager(inputs=self.inputs,
                                                  target=self.target)

    def tearDown(self):
        pass

    def test_sklearn_dataset_01_init(self):
        """ Testing dataset getter method """

        # Test if arguments provided as pd.DataFrame and they have been properly converted to the
        # np.ndarray
        inputs = pd.DataFrame(self.inputs)
        target = pd.DataFrame(self.target)
        self.sklearn_dataset = SkLearnDataManager(inputs=inputs,
                                                  target=target)
        self.assertIsInstance(self.sklearn_dataset._inputs, np.ndarray)
        self.assertIsInstance(self.sklearn_dataset._target, np.ndarray)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
