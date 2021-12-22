# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import unittest

import copy
from random import random

import torch
from torch.nn import Linear
import numpy as np
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


    def test_fed_average_torch(self):

        aggregator = FedAverage()
        aggregated_params = aggregator.aggregate(self.models, self.weights)
        # ===============================================================
        # Assert Federated Average
        # ===============================================================
        for key, val in aggregated_params.items():
            self.assertTrue( torch.isclose(val, self.model.state_dict()[key]).all())


    def test_fed_average_sklearn_sgd_t1(self):
        model_params = [{'coef_': np.array([3, 8, 8, 3, 1]), 'intercept_': np.array([4])},
                        {'coef_': np.array([0.4, 1.6, 2, 1, 0.1]), 'intercept_': np.array([1])},
                        {'coef_': np.array([2 ,5, 5, 3, 1]), 'intercept_': np.array([6])}]
        weights = [0.2, 0.2, 0.6]
        aggregator = FedAverage()
        aggregated_params = aggregator.aggregate(model_params, weights)
        self.assertTrue( np.allclose( aggregated_params['coef_'] , np.array( [1.88 , 4.92 , 5. , 2.6 , 0.82] ) ) )
        self.assertTrue( np.allclose(aggregated_params['intercept_'], np.array( [4.6] )))


    def test_fed_average_sklearn_sgd_t2(self):
        weights = [0.27941176470588236, 0.7205882352941176]

        model_params = [{'coef_':np.array([-0.02629813 , 0.04612957 ,-0.00321454,  0.08003535 , 0.30818439]), 'intercept_':  np.array([0.161345])},
                        {'coef_':np.array([-0.02782622 , 0.0145883 , -0.01471519 ,-0.03673147 , 0.45426254] ), 'intercept_':  np.array([-0.00457364])}]

        aggregator = FedAverage()
        aggregated_params = aggregator.aggregate(model_params, weights)
        self.assertTrue(np.allclose(aggregated_params['coef_'], np.array([-0.02739925,  0.0234013,  -0.01150177, -0.00410545 , 0.41344659] )))
        self.assertTrue(np.allclose(aggregated_params['intercept_'], np.array( [0.04178598] )))

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
