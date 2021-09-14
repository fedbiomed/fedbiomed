import unittest

import copy
from random import random
from joblib import dump, load
import torch
from torch.nn import Linear
import numpy as np
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from sklearn.linear_model import SGDRegressor
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
import os


class TestModel(SGDSkLearnModel):

    def adhoc(self):
        print('adhoc')

class TestFedbiosklearn(unittest.TestCase):

    def setUp(self):
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)


    def tearDown(self):
        pass

    def test_init(self):
        kw = {'toto': 'le' , 'lelo':'la', 'max_iter':7000, 'tol': 0.3456, 'n_features': 10, 'model': 'SGDRegressor' }
        fbsk = SGDSkLearnModel(kw)
        m = fbsk.get_model()
        p = m.get_params()
        self.assertEqual(p['max_iter'] , 7000)
        self.assertEqual(p['tol'],  0.3456 )
        self.assertTrue( np.allclose(m.coef_, np.zeros(10)) )
        self.assertIsNone(p.get('lelo'))
        self.assertIsNone(p.get('toto'))
        self.assertIsNone(p.get('model'))

    def test_not_implemented_method(self):
        kw = {'toto': 'le', 'lelo': 'la', 'max_iter': 7000, 'tol': 0.3456, 'n_features': 10, 'model': 'SGDRegressor'}
        t = TestModel(kw)
        self.assertRaises(NotImplementedError,lambda: t.training_data())


    def test_save_and_load(self):
        CURRENTDIR = os.path.abspath(os.path.join(__file__, os.pardir))
        print('curdir ',CURRENTDIR)
        filename = os.path.join(CURRENTDIR,'sgd.sav')
        skm = SGDSkLearnModel({'max_iter': 1000, 'tol':1e-3, 'n_features': 5, 'model': 'SGDRegressor'})
        skm.save(filename)

        self.assertTrue(os.path.exists(filename) and os.path.getsize(filename) > 0  )

        m = skm.load(filename)

        self.assertEqual(m.max_iter,1000)
        self.assertEqual(m.tol, 0.001)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
