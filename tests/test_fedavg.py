import unittest

import copy
from random import random

import torch
from torch.nn import Linear

from fedbiomed.researcher.aggregators.fedavg import FedAverage

class TestFedaverage(unittest.TestCase):
    '''
    Test the Fedaverage class
    '''
    # before the tests
    def setUp(self):
        self.model = Linear(10, 3)
        self.models = [copy.deepcopy(self.model).state_dict() for _ in range(4)]
        self.weights = [random() for _ in self.models]

    # after the tests
    def tearDown(self):
        pass


    def test_fed_average(self):

        aggregator = FedAverage()
        aggregated_params = aggregator.aggregate(self.models, self.weights)

        # ===============================================================
        # Assert Federated Average
        # ===============================================================
        for key, val in aggregated_params.items():
            self.assertTrue( torch.isclose(val, self.model.state_dict()[key]).all())



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
