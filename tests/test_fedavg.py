from testsupport.base_case import ResearcherTestCase

import copy
from random import random
import unittest
import torch
from torch.nn import Linear
import numpy as np


from fedbiomed.researcher.aggregators.fedavg import FedAverage



class TestFedaverage(ResearcherTestCase):
    '''
    Test the FedAverage class
    '''

    # before the tests
    def setUp(self):
        self.model = Linear(10, 3)
        self.models = [{f'node_{i}': copy.deepcopy(self.model).state_dict()} for i in range(4)]
        self.weights = [random() for _ in self.models]
        self.aggregator = FedAverage()


    # after the tests
    def tearDown(self):
        pass

    def test_fed_average_01_torch(self):
        """ Testing aggregation for torch model """

        aggregated_params = self.aggregator.aggregate(self.models, self.weights)
        # ===============================================================
        # Assert Federated Average
        # ===============================================================
        for key, val in aggregated_params.items():
            self.assertTrue(torch.isclose(val, self.model.state_dict()[key]).all())

    def test_fed_average_02_sklearn_sgd_t1(self):
        """ Testing aggregation for sklearn sgd test 1"""

        model_params = [{'node_1':{'coef_': np.array([3, 8, 8, 3, 1]), 'intercept_': np.array([4])}},
                        {'node_2': {'coef_': np.array([0.4, 1.6, 2, 1, 0.1]), 'intercept_': np.array([1])}},
                        {'mode_3': {'coef_': np.array([2, 5, 5, 3, 1]), 'intercept_': np.array([6])}}]
        weights = [0.2, 0.2, 0.6]

        aggregated_params = self.aggregator.aggregate(model_params, weights)
        self.assertTrue(np.allclose(aggregated_params['coef_'], np.array([1.88, 4.92, 5., 2.6, 0.82])))
        self.assertTrue(np.allclose(aggregated_params['intercept_'], np.array([4.6])))

    def test_fed_average_03_sklearn_sgd_t2(self):
        """ Testing aggregation for sklearn sgd test 2"""

        weights = [0.27941176470588236, 0.7205882352941176]

        model_params = [{'node_1': {'coef_': np.array([-0.02629813, 0.04612957, -0.00321454, 0.08003535, 0.30818439]),
                         'intercept_': np.array([0.161345])}},
                        {'node_2': {'coef_': np.array([-0.02782622, 0.0145883, -0.01471519, -0.03673147, 0.45426254]),
                         'intercept_': np.array([-0.00457364])}}]

        aggregated_params = self.aggregator.aggregate(model_params, weights)
        self.assertTrue(np.allclose(aggregated_params['coef_'],
                                    np.array([-0.02739925, 0.0234013, -0.01150177, -0.00410545, 0.41344659])))
        self.assertTrue(np.allclose(aggregated_params['intercept_'], np.array([0.04178598])))

    def test_fed_average_04_save_state(self):
        """ Testing FedAverage save state """

        expected_state = {'class': 'FedAverage',
                          'module': 'fedbiomed.researcher.aggregators.fedavg',
                          'parameters': None}
        state = self.aggregator.save_state()
        self.assertDictEqual(expected_state, state, 'State of FedAvg has not been saved correctly')

    def test_fed_average_05_load_state(self):

        """ Testing FedAverage load state """

        state = {
            'parameters': {'param': True}
        }
        self.aggregator.load_state(state)
        self.assertDictEqual(self.aggregator._aggregator_args,
                             state['parameters'],
                             'The state of the aggregator class has not been loaded correctly')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
