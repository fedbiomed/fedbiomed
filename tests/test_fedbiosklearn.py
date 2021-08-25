import unittest

import copy
from random import random
from joblib import dump, load
import torch
from torch.nn import Linear
import numpy as np
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from sklearn.linear_model import SGDRegressor
from fedbiomed.common.fedbiosklearn import SkLearnModel
import os
class TestFedbiosklearn(unittest.TestCase):

    def setUp(self):
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)


    def tearDown(self):
        pass

    def test_init(self):
        kw = {'toto': 'le' , 'lelo':'la', 'max_iter':7000, 'tol': 0.3456, 'number_columns': 10, 'model': 'Kmeans' }
        fbsk = SkLearnModel(kw)
        m = fbsk.get_model()
        p = m.get_params()
        self.assertEqual(p['max_iter'] , 7000)
        self.assertEqual(p['tol'],  0.3456 )
        self.assertTrue( np.allclose(m.coef_, np.zeros(10)) )
        self.assertIsNone(p.get('lelo'))
        self.assertIsNone(p.get('toto'))
        self.assertIsNone(p.get('model'))



    def test_save_and_load(self):
        CURRENTDIR =  os.getcwd()
        print('curdir ',CURRENTDIR)
        skm = SkLearnModel({'max_iter': 1000, 'tol':1e-3, 'number_columns': 5})
        skm.save(CURRENTDIR + '/tests/data/sgd.sav')

        self.assertTrue(os.path.exists(CURRENTDIR + '/tests/data/sgd.sav') and os.path.getsize(CURRENTDIR + '/tests/data/sgd.sav') > 0  )

        m = skm.load(CURRENTDIR + '/tests/data/sgd.sav')

        self.assertEqual(m.max_iter,1000)
        self.assertEqual(m.tol, 0.001)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
