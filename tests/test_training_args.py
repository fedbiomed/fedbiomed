"""
Test module for TrainingArgs
"""


import unittest

from fedbiomed.common.exceptions import FedbiomedUserInputError
from fedbiomed.common.metrics import MetricTypes
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

    def test_training_args_01_init(self):
        """
        simple initialisation tests
        """
        t = TrainingArgs( { "lr": 2.0, "epochs": 1 } )
        self.assertEqual( t['lr'], 2.0)
        self.assertEqual( t['epochs'], 1)
        with self.assertRaises( FedbiomedUserInputError ):
            print( t['dry_run'] )

        t = TrainingArgs( only_required = False )

        # lr has no default value
        with self.assertRaises(FedbiomedUserInputError):
            t.default_value('lr')

        # and test_ratio has
        self.assertEqual( t['test_ratio'], t.default_value('test_ratio'))

        # init with bad given values
        with self.assertRaises(FedbiomedUserInputError):
            t = TrainingArgs( { "lr": 2.0, "epochs": "not_an_int" } )


    def test_training_args_02_scheme(self):
        """
        play with schemes
        """
        t = TrainingArgs()
        self.assertEqual( t.scheme(), TrainingArgs.default_scheme() )

        my_added_rule = {
            'fun': {
                "rules": [bool],
                "required": True,
                "default": True
            }
        }
        t = TrainingArgs( extra_scheme = my_added_rule)

        my_wrong_added_rule = {
            'wrong': {
            }
        }
        with self.assertRaises(FedbiomedUserInputError):
            t = TrainingArgs( extra_scheme = my_wrong_added_rule)

        my_no_default_added_rule = {
            'wrong_no_default': {
                "rules": [int],
                "required": True
            }
        }

        with self.assertRaises(FedbiomedUserInputError):
            t = TrainingArgs( extra_scheme = my_no_default_added_rule)

    def test_training_args_03_setters(self):
        """
        test TrainingArgs setters
        """
        t = TrainingArgs( only_required = False )

        t ^= { "lr": 3.14 }
        self.assertEqual( t['lr'], 3.14)

        # test all library hooks for setters
        with self.assertRaises(FedbiomedUserInputError):
            t ^= { 'lr': -1.0 }

        with self.assertRaises(FedbiomedUserInputError):
            t ^= { 'test_ratio': -1.0 }

        with self.assertRaises(FedbiomedUserInputError):
            t ^= { 'does_not_exist' : "what_else_?" }

        # test getters
        with self.assertRaises(FedbiomedUserInputError):
            print( t['does_not_exist'] )

        # default value getter
        with self.assertRaises(FedbiomedUserInputError):
            print( t.default_value('lr') )

        with self.assertRaises(FedbiomedUserInputError):
            print( t.default_value('does_not_exist') )

        # dict() getter
        dico = t.dict()
        self.assertEqual( dico['lr'], 3.14)

        # get() getter
        self.assertEqual( t.get('lr'), 3.14)
        self.assertEqual( t.get('this_one_is_stupid'), None)
        self.assertEqual( t.get('should_we_keep_the_get_method_?', False), False)

        # how to properly test __repr__ and __str__ ?
        t_to_string = str(t)
        for s in [ '{', '}', 'test_ratio', 'lr']:
            self.assertIn( s, t_to_string)

        t_to_string = t.__repr__()
        for s in [ '{', '}', 'test_ratio', 'lr']:
            self.assertIn( s, t_to_string)

    def test_training_args_04_metric(self):
        """
        test metric validator
        """
        t = TrainingArgs()

        # test_metric key
        t ^= { "test_metric": None }
        self.assertEqual( t['test_metric'], None)

        t ^= { "test_metric": "ACCURACY" }
        self.assertEqual( t['test_metric'], "ACCURACY")

        t ^= { "test_metric": MetricTypes.ACCURACY }
        self.assertEqual( t['test_metric'], MetricTypes.ACCURACY)

        dico = t.dict()
        self.assertEqual( dico['test_metric'], "ACCURACY")

        with self.assertRaises(FedbiomedUserInputError):
            t ^= { "test_metric": "RULE_OF_THUMB" }
        # t was not changed by previous line
        self.assertEqual( t['test_metric'], MetricTypes.ACCURACY)

        # test_metric_args key
        t ^= { "test_metric_args" : { "must_be_a_dict": True} }

        with self.assertRaises(FedbiomedUserInputError):
            t ^= { "test_metric_args": "not a dict" }


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
