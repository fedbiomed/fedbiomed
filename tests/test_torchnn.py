# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import unittest
import os
import sys

import torch
import torch.nn as nn

from fedbiomed.common.torchnn import TorchTrainingPlan

# define TP outside of test class to avoid indentation problems when exporting class to file
class TrainingPlan(TorchTrainingPlan):
    def __init__(self):
        super(TrainingPlan, self).__init__()
        self.lin1 = nn.Linear(4,2)
        
    def test_method(self):
        return True

class TestTorchnn(unittest.TestCase):
    """
    Test the Torchnn class
    """
    # before the tests
    def setUp(self):
        self.TrainingPlan = TrainingPlan
        self.params = { 'one': 1, '2': 'two' }
        self.tmpdir = '.'

    # after the tests
    def tearDown(self):
        pass

    #
    # TODO : add tests for checking the training payload
    #

    def test_save_load_model(self):

        tp1 = self.TrainingPlan()
        self.assertIsNotNone(tp1.test_method)
        self.assertTrue(tp1.test_method())
        
        modulename = 'tmp_model'
        codefile = self.tmpdir + os.path.sep + modulename + '.py'
        try:
            os.remove(codefile)
        except FileNotFoundError:
            pass

        tp1.save_code(codefile)
        self.assertTrue(os.path.isfile(codefile))

        # would expect commented lines to be necessary
        #
        #sys.path.insert(0, self.tmpdir)
        #exec('import ' + modulename, globals())
        exec('import ' + modulename)
        #sys.path.pop(0)
        TrainingPlan2 = eval(modulename + '.' + self.TrainingPlan.__name__)
        tp2 = TrainingPlan2()

        self.assertIsNotNone(tp2.test_method)
        self.assertTrue(tp2.test_method())

        os.remove(codefile)
        
    def test_save_load_params(self):
        tp1 = TrainingPlan()
        paramfile = self.tmpdir + '/tmp_params.pt'
        try:
            os.remove(paramfile)
        except FileNotFoundError:
            pass

        # save/load from/to variable
        tp1.save(paramfile, self.params)
        self.assertTrue(os.path.isfile(paramfile))
        params2 = tp1.load(paramfile, True)

        self.assertTrue(type(params2) is dict)
        self.assertEqual(self.params, params2)

        # save/load from/to object params
        tp1.save(paramfile)
        tp2 = TrainingPlan()
        tp2.load(paramfile)
        self.assertTrue(type(params2) is dict)

        sd1 = tp1.state_dict()
        sd2 = tp2.state_dict()

        # verify we have an equivalent state dict
        for key in sd1:
            self.assertTrue(key in sd2)

        for key in sd2:
            self.assertTrue(key in sd1)

        for (key, value) in sd1.items():
            self.assertTrue(torch.all(torch.isclose(value, sd2[key])))

        os.remove(paramfile)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()


