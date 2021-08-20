import unittest
import os
import sys

from fedbiomed.common.torchnn import TorchTrainingPlan

# define outside of test class to avoid indentation problems when exporting class to file
class TrainingPlan(TorchTrainingPlan):
    def __init__(self):
        super(TrainingPlan, self).__init__()
        
    def test_method(self):
        return True

class TestTorchnn(unittest.TestCase):
    """
    Test the Torchnn class
    """
    # before the tests
    def setUp(self):
        self.TrainingPlan = TrainingPlan

    # after the tests
    def tearDown(self):
        pass

    # TODO : add tests for training payload

    def test_save_load(self):
        tp1 = self.TrainingPlan()
        self.assertIsNotNone(tp1.test_method)
        self.assertTrue(tp1.test_method())
        
        modulename = 'tmp_model'
        tmpdir = '.'
        codefile = tmpdir + '/' + modulename + '.py'
        #paramfile = 'tmp_param.pt'

        try:
            os.remove(codefile)
        except FileNotFoundError:
            pass
        tp1.save_code(codefile)
        self.assertTrue(os.path.isfile(codefile))

        # would expect commented lines to be necessary
        #
        #sys.path.insert(0, tmpdir)
        #exec('import ' + modulename, globals())
        exec('import ' + modulename)
        #sys.path.pop(0)
        TrainingPlan2 = eval(modulename + '.' + self.TrainingPlan.__name__)
        tp2 = TrainingPlan2()

        self.assertIsNotNone(tp2.test_method)
        self.assertTrue(tp2.test_method())

        try:
            os.remove(codefile)
        except FileNotFoundError:
            pass
        
if __name__ == '__main__':  # pragma: no cover
    unittest.main()


