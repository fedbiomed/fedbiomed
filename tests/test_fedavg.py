from testsupport.base_case import ResearcherTestCase

import copy
from random import random, shuffle
import unittest
from fedbiomed.common.exceptions import FedbiomedAggregatorError

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
        self.models = {f'node_{i}': copy.deepcopy(self.model).state_dict() for i in range(4)}
        self.weights = {f'node_{i}': random() for i, _ in enumerate(self.models)}
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

        model_params = {'node_1': {'coef_': np.array([3, 8, 8, 3, 1]), 'intercept_': np.array([4])},
                        'node_2': {'coef_': np.array([0.4, 1.6, 2, 1, 0.1]), 'intercept_': np.array([1])},
                        'node_3': {'coef_': np.array([2, 5, 5, 3, 1]), 'intercept_': np.array([6])}
                        }
        weights = {
            'node_1': 0.2,
            'node_2': 0.2,
            'node_3': 0.6
        }

        aggregated_params = self.aggregator.aggregate(model_params, weights)
        self.assertTrue(np.allclose(aggregated_params['coef_'], np.array([1.88, 4.92, 5., 2.6, 0.82])))
        self.assertTrue(np.allclose(aggregated_params['intercept_'], np.array([4.6])))

    def test_fed_average_03_sklearn_sgd_t2(self):
        """ Testing aggregation for sklearn sgd test 2"""

        weights = {
            'node_1': 0.27941176470588236,
            'node_2': 0.7205882352941176
        }

        model_params = {
            'node_1': {'coef_': np.array([-0.02629813, 0.04612957, -0.00321454, 0.08003535, 0.30818439]),
                       'intercept_': np.array([0.161345])},
            'node_2': {'coef_': np.array([-0.02782622, 0.0145883, -0.01471519, -0.03673147, 0.45426254]),
                       'intercept_': np.array([-0.00457364])}
        }

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

    def test_fed_average_06_order_of_weight_and_model_params(self):
        """Tests bug #433 where weights and model params have scrambled order."""

        # first we are testing with sklearn framework
        weights = {
            'node_7ea1c779': 0.9,
            'node_02a6e376': 0.1,
        }
        model_params = {'node_02a6e376': {'coef_': np.array([0.])},
                        'node_7ea1c779': {'coef_': np.array([10.])}}
        agg_params = self.aggregator.aggregate(model_params=model_params,
                                               weights=weights)
        self.assertEqual(agg_params['coef_'][0], 9.)

        # test if missing node id triggers exception
        model_params['node_123abc'] = {'coef_': np.array([5.])}

        with self.assertRaises(FedbiomedAggregatorError):
            self.aggregator.aggregate(model_params=model_params,
                                      weights=weights)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
