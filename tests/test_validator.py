import unittest
from fedbiomed.common.validator import Validator, validator_decorator


class TestValidator(unittest.TestCase):
    '''
    Test the Validator class
    '''

    # before every tests
    def setUp(self):
        pass

    # after every tests
    def tearDown(self):
        pass

    # define some validation hooks to add to the validator
    @staticmethod
    @validator_decorator
    def hook_01_positive_integer_check(value):
        """
        value must be a positive integer
        """
        if not isinstance(value, int) or value < 1:
            return False
        return True

    @staticmethod
    def hook_02_positive_integer_check(value):
        """
        value must be a positive integer

        if we do not use the decorator, we should return a tuple
        (boolean, error_message_string)
        """
        if not isinstance(value, int) or value < 1:
            return False, str(value) + " is not a positive integer"
        return True, None


    @staticmethod
    @validator_decorator
    def hook_probability_check(value):
        """
        float between 0.0 and 1.0 expected
        """
        if not isinstance(value, float) or value < 0.0 or value > 1.0:
            return False, "float between [ 0.0, 1.0 ] expected"
        return True

    def test_validator_01_typechecking(self):
        """
        simple and direct type checking tests
        """
        self.assertTrue( Validator().validate(1, int))
        self.assertTrue( Validator().validate(1.0, float))
        self.assertTrue( Validator().validate( {} , dict ))
        self.assertTrue( Validator().validate( { "un": 1 } , dict ))
        self.assertTrue( Validator().validate(1, int))


    def test_validator_02_use_function_directly(self):
        self.assertTrue( Validator().validate(1, self.hook_01_positive_integer_check))
        self.assertFalse( Validator().validate(-1, self.hook_01_positive_integer_check))
        self.assertFalse( Validator().validate(1.0, self.hook_01_positive_integer_check))

    def test_validator_03_registration(self):

        v = Validator()

        rule_name = 'rule_01'

        # rule is unknown
        self.assertFalse( v.knows_rule(rule_name) )

        # must be None
        self.assertTrue( v.rule(rule_name) is None)

        # register the rule
        self.assertTrue(v.register_rule( rule_name, self.hook_01_positive_integer_check))

        # must be known
        self.assertTrue( v.knows_rule(rule_name) )

        # use the registered hook
        self.assertTrue( Validator().validate(1, rule_name))
        self.assertFalse( Validator().validate(-1, rule_name))
        self.assertFalse( Validator().validate(1.0, rule_name))

        # must be the registered function
        rule = v.rule( rule_name)
        self.assertEqual(rule.__name__, self.hook_01_positive_integer_check.__name__ )

        # unregister
        v.delete_rule(rule_name)
        self.assertFalse( v.knows_rule(rule_name) )

        # register several time the same rule
        self.assertTrue(v.register_rule( rule_name, self.hook_01_positive_integer_check))
        self.assertFalse(v.register_rule( rule_name, self.hook_01_positive_integer_check))
        self.assertTrue(v.register_rule( rule_name,
                         self.hook_01_positive_integer_check,
                         override = True))

    def test_validator_04_another_one(self):

        v = Validator()
        self.assertTrue(v.register_rule( 'probability', self.hook_probability_check))

        self.assertFalse( v.validate( "un quart", 'probability' ) )

        self.assertFalse( v.validate( -1.0, 'probability' ) )
        self.assertFalse( v.validate( -0.00001, 'probability' ) )

        self.assertTrue( v.validate( 0.0, 'probability' ) )
        self.assertTrue( v.validate( 0.25, 'probability' ) )
        self.assertTrue( v.validate( 0.75, 'probability' ) )
        self.assertTrue( v.validate( 1.0, 'probability' ) )

        self.assertFalse( v.validate( 1.00001, 'probability' ) )
        self.assertFalse( v.validate( 7.0, 'probability' ) )


    def test_validator_05_without_decorator(self):
        v = Validator()

        rule_name = 'rule_02'

        # register the rule
        self.assertTrue(v.register_rule( rule_name, self.hook_02_positive_integer_check))

        # checks
        self.assertTrue( Validator().validate(1, rule_name))
        self.assertFalse( Validator().validate(-1, rule_name))
        self.assertFalse( Validator().validate(1.0, rule_name))


    def test_validator_06_strict_or_not(self):

        v = Validator()

        rule_name = 'this_rule_is_unknown'

        self.assertFalse( v.knows_rule(rule_name) )
        self.assertFalse( v.validate( 0, rule_name))
        self.assertTrue( v.validate( 0, rule_name, strict = False))


    @staticmethod
    @validator_decorator
    def loss_rate_validation_hook(value):
        """
        float between 0.0 and 1.0
        """
        if not isinstance(value, float) or value < 0.0 or value > 1.0:
            return False, "float between [ 0.0, 1.0 ] expected"
        return True

    def test_validator_07_use_scheme(self):

        # training_args must be a dict()
        # and contains the required 'lr' field
        # 'lr' value is checked against 2 rules
        training_args_scheme = {
            'lr' : { 'rules': [ float, self.loss_rate_validation_hook] ,
                     'required': True,
                     'default': 1.0
                    },
        }

        self.assertFalse( Validator().validate(
            {} ,
            training_args_scheme ) )

        self.assertTrue( Validator().validate(
            { 'lr' : 0.4 } ,
            training_args_scheme ) )

        self.assertFalse( Validator().validate(
            { 'lr' : 0.4 , 'extra': "extra field"} ,
            training_args_scheme ) )

        # same, but lr is not required
        training_args_scheme = {
            'lr' : { 'rules': [ float, self.loss_rate_validation_hook] ,
                     'default': 1.0
                    },
        }

        self.assertTrue( Validator().validate(
            {} ,
            training_args_scheme ) )

        self.assertTrue( Validator().validate(
            { 'lr' : 0.4 } ,
            training_args_scheme ) )

        self.assertFalse( Validator().validate(
            { 'lr' : 0.4 , 'extra': "extra field"} ,
            training_args_scheme ) )

        # same again
        training_args_scheme = {
            'lr' : { 'rules': [ float, self.loss_rate_validation_hook] ,
                     'default': 1.0,
                     'required': False
                    },
        }

        self.assertTrue( Validator().validate(
            {} ,
            training_args_scheme ) )

        self.assertTrue( Validator().validate(
            { 'lr' : 0.4 } ,
            training_args_scheme ) )

        self.assertFalse( Validator().validate(
            { 'lr' : 0.4 , 'extra': "extra field"} ,
            training_args_scheme ) )

    def test_validator_08_validate_the_validator(self):

        training_args_scheme = {
            'lr' : { 'rules': [ float, self.loss_rate_validation_hook] ,
                     'default': 1.0
                    },
        }

        v = Validator()
        self.assertTrue(v.register_rule( "tr_01", training_args_scheme))

        training_arg_scheme = {
            'lr' : { 'rules': [ self.loss_rate_validation_hook ] ,
                     'required': True,
                     'unallowed_key': False
                    },
        }

        # should be False !
        self.assertTrue(v.register_rule( "tr_02", training_args_scheme))
        self.assertTrue( Validator().validate(
            { 'lr' : 0.4 } ,
            training_args_scheme ) )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
