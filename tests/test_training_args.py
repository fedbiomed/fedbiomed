import unittest
from fedbiomed.common.training_args import TrainingArgs


class TestTrainingArgs(unittest.TestCase):
    '''
    Test the TrainingArgs class
    '''

    # before every test
    def setUp(self):
        pass

    # after every test
    def tearDown(self):
        pass

    def test_training_args_01_simple(self):
        """
        simple initialisation tests
        """
        t = TrainingArgs( { "lr": 2.0, "epochs": 1 } )


    def test_training_args_02_setters(self):
        """
        test TrainingArgs setters
        """

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
