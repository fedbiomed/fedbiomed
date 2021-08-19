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

    def test_save_and_load(self):
        CURRENTDIR =  os.getcwd()
        print('curdir ',CURRENTDIR)
        skm = SkLearnModel({'max_iter': 1000, 'tol':1e-3})
        skm.save(CURRENTDIR + '/tests/data/sgd.sav')

        self.assertTrue(os.path.exists(CURRENTDIR + '/tests/data/sgd.sav') and os.path.getsize(CURRENTDIR + '/tests/data/sgd.sav') > 0  )

        m = skm.load(CURRENTDIR + '/tests/data/sgd.sav')

        self.assertEqual(m.max_iter,1000)
        self.assertEqual(m.tol, 0.001)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
