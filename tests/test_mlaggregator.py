import unittest

import copy
from random import random

import numpy as np
from fedbiomed.researcher.aggregators.ppca_aggregator import MLaggregator

class TestMLaggregator(unittest.TestCase):
    '''
    Test the MLaggregator class
    '''
    # before the tests
    def setUp(self):
        K = 3
        D_i = [3, 4, 5]
        q_gen, sigma2_gen1, sigma2_gen2, sigma2_gen3 = 2, 0.2, 0.1, 0.3
        W_gen1 = np.random.uniform(-1.5, 1.5, (D_i[0], q_gen)).reshape((D_i[0], q_gen))
        W_gen2 = np.random.uniform(-0.5, 0.5, (D_i[1], q_gen)).reshape((D_i[1], q_gen))
        W_gen3 = np.random.uniform(-2, 2, (D_i[2], q_gen)).reshape((D_i[2], q_gen))
        mu_gen1 = np.random.uniform(-1, 1, D_i[0]).reshape((D_i[0], 1))
        mu_gen2 = np.random.uniform(-0.5, 0.5, D_i[1]).reshape((D_i[1], 1))
        mu_gen3 = np.random.uniform(-2, 2, D_i[2]).reshape((D_i[2], 1))
        Wk = [W_gen1, W_gen2, W_gen3]
        muk = [mu_gen1, mu_gen2, mu_gen3]
        sigma2k = [sigma2_gen1, sigma2_gen2, sigma2_gen3]
        self.model = {'K': K,
                      'views_names': range(K),
                      'dimensions': (D_i,q_gen),
                      'Wk': Wk, 
                      'muk': muk,
                      'sigma2k': sigma2k}
        self.models = [copy.deepcopy(self.model) for _ in range(4)]
        self.weights = [random() for _ in self.models]

    # after the tests
    def tearDown(self):
        pass


    def test_mlaggregator(self):
        K = 3
        D_i = [3, 4, 5]
        q_gen, sigma2_gen1, sigma2_gen2, sigma2_gen3 = 2, 0.2, 0.1, 0.3
        W_gen1 = np.random.uniform(-1.5, 1.5, (D_i[0], q_gen)).reshape((D_i[0], q_gen))
        W_gen2 = np.random.uniform(-0.5, 0.5, (D_i[1], q_gen)).reshape((D_i[1], q_gen))
        W_gen3 = np.random.uniform(-2, 2, (D_i[2], q_gen)).reshape((D_i[2], q_gen))
        mu_gen1 = np.random.uniform(-1, 1, D_i[0]).reshape((D_i[0], 1))
        mu_gen2 = np.random.uniform(-0.5, 0.5, D_i[1]).reshape((D_i[1], 1))
        mu_gen3 = np.random.uniform(-2, 2, D_i[2]).reshape((D_i[2], 1))
        Wk = [W_gen1, W_gen2, W_gen3]
        muk = [mu_gen1, mu_gen2, mu_gen3]
        sigma2k = [sigma2_gen1, sigma2_gen2, sigma2_gen3]
        model_param = {'K': K,
                      'dimensions': (D_i,q_gen),
                      'views_names': range(K),
                      'Wk': Wk, 
                      'muk': muk,
                      'sigma2k': sigma2k}
        model_params = [copy.deepcopy(model_param) for _ in range(4)]
        weights = [random() for _ in self.models]
        aggregator = MLaggregator()
        aggregated_params = aggregator.aggregate(model_params, weights)
        for k in range(K):
            self.assertTrue( np.allclose( aggregated_params['tilde_muk'][k] , model_param['muk'][k] ) )
            self.assertTrue( np.allclose( aggregated_params['tilde_Wk'][k] , model_param['Wk'][k] ) )
            self.assertTrue( np.allclose( np.array(aggregated_params['tilde_Sigma2k'][k]) , np.array(model_param['sigma2k'][k]) ) )

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
