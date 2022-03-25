import os
import tempfile
import unittest

import numpy as np
from sklearn.linear_model import SGDRegressor

from fedbiomed.common.training_plans import SGDSkLearnModel
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError


class TestFedbiosklearn(unittest.TestCase):

    def setUp(self):
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)

    def tearDown(self):
        pass

    def test_init(self):
        kw = {'toto': 'le', 'lelo': 'la', 'max_iter': 7000, 'tol': 0.3456, 'n_features': 10, 'model': 'SGDRegressor'}
        fbsk = SGDSkLearnModel(kw)
        m = fbsk.get_model()
        p = m.get_params()
        self.assertEqual(p['max_iter'], 7000)
        self.assertEqual(p['tol'], 0.3456)
        self.assertTrue(np.allclose(m.coef_, np.zeros(10)))
        self.assertIsNone(p.get('lelo'))
        self.assertIsNone(p.get('toto'))
        self.assertIsNone(p.get('model'))

    def test_save_and_load(self):
        randomfile = tempfile.NamedTemporaryFile()

        skm = SGDSkLearnModel({'max_iter': 1000, 'tol': 1e-3, 'n_features': 5, 'model': 'SGDRegressor'})
        skm.save(randomfile.name)

        self.assertTrue(os.path.exists(randomfile.name) and os.path.getsize(randomfile.name) > 0)

        m = skm.load(randomfile.name)

        self.assertEqual(m.max_iter, 1000)
        self.assertEqual(m.tol, 0.001)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
