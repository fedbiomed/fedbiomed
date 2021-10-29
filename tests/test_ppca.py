import tempfile
import os
import unittest

import numpy as np

from fedbiomed.researcher.aggregators.ppca_aggregator import MLaggregator
from fedbiomed.common.ppca import PpcaPlan


class TestModel(PpcaPlan):
    """
    What is it ?
    """
    def adhoc(self):
        print('adhoc')

class TestPpca(unittest.TestCase):

    def setUp(self):
        self.model = {'K': None,
                      'dimensions': None,
                      # local params:
                      'Wk': None, 
                      'muk': None,
                      'sigma2k': None}

    def tearDown(self):
        pass

    def test_init(self):
        kw = {'tot_views': 3, 'dim_views': [3, 4, 5], 'n_components': 2, 'is_norm': True}
        ppca = PpcaPlan(kw)
        p = ppca.get_model()
        self.assertEqual(p['K'] , 3)
        self.assertEqual(p['dimensions'],  ([3, 4, 5],2) )
        self.assertIsNone(p['muk'])
        self.assertIsNone(p['Wk'])
        self.assertIsNone(p['sigma2k'])

    def test_save_and_load(self):
        randomfile = tempfile.NamedTemporaryFile()

        mvppca = PpcaPlan({'tot_views': 3, 'dim_views': [3, 4, 5], 'n_components': 2, 'is_norm': True})
        mvppca.save(randomfile.name)

        self.assertTrue(os.path.exists(randomfile.name) and os.path.getsize(randomfile.name) > 0  )

        p = mvppca.load(randomfile.name)

        self.assertEqual(p['dimensions'],  ([3, 4, 5],2) )

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
