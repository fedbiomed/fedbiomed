import unittest
from fedbiomed.common.validator import Validator, SchemeValidator, validator_decorator, _ValidatorHookType


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


    def test_validator_00_allowed_hook_types(self):
        """
        check all authorized hook types
        """

        # all type checking
        self.assertTrue(Validator._is_hook_type_valid( bool ))
        self.assertTrue(Validator._is_hook_type_valid( int ))
        self.assertTrue(Validator._is_hook_type_valid( float ))
        self.assertTrue(Validator._is_hook_type_valid( str ))
        self.assertTrue(Validator._is_hook_type_valid( list ))
        self.assertTrue(Validator._is_hook_type_valid( dict ))

        # function
        self.assertTrue(Validator._is_hook_type_valid( self.hook_probability_check ))

        # scheme
        self.assertTrue(Validator._is_hook_type_valid( {} ))
        scheme = {
            'a' : { 'rules': [ float ] }
        }
        sc = SchemeValidator(scheme)
        self.assertTrue(Validator._is_hook_type_valid( sc ))

        # unallowed
        self.assertFalse(Validator._is_hook_type_valid( [] ))
        self.assertFalse(Validator._is_hook_type_valid( 3.14 ))
        self.assertFalse(Validator._is_hook_type_valid( "bad_entry" ))


    def test_validator_01_typechecking(self):
        """
        simple and direct type checking tests
        """
        self.assertTrue( Validator().validate(True, bool))
        self.assertTrue( Validator().validate(1, int))
        self.assertTrue( Validator().validate(1.0, float))
        self.assertTrue( Validator().validate( {} , dict ))
        self.assertTrue( Validator().validate( { "un": 1 } , dict ))
        self.assertTrue( Validator().validate( [], list))
        self.assertTrue( Validator().validate( "one", str))

    def test_validator_02_use_function_directly(self):
        self.assertTrue( Validator().validate(1, self.hook_01_positive_integer_check))
        self.assertFalse( Validator().validate(-1, self.hook_01_positive_integer_check))
        self.assertFalse( Validator().validate(1.0, self.hook_01_positive_integer_check))

    def test_validator_03_registration(self):

        v = Validator()

        rule_name = 'rule_positive_integer'

        # rule is unknown
        self.assertFalse( v.is_known_rule(rule_name) )

        # must be None
        self.assertTrue( v.rule(rule_name) is None)

        # register the rule
        self.assertTrue(v.register( rule_name, self.hook_01_positive_integer_check))

        # must be known
        self.assertTrue( v.is_known_rule(rule_name) )

        # use the registered hook
        self.assertTrue( Validator().validate(1, rule_name))
        #import pdb; pdb.set_trace()
        self.assertFalse( Validator().validate(-1, rule_name))
        self.assertFalse( Validator().validate(1.0, rule_name))

        # must be the registered function
        rule = v.rule( rule_name)
        self.assertEqual(rule.__name__, self.hook_01_positive_integer_check.__name__ )

        # unregister
        v.delete(rule_name)
        self.assertFalse( v.is_known_rule(rule_name) )

        # register several time the same rule
        self.assertTrue(v.register( rule_name, self.hook_01_positive_integer_check))
        self.assertFalse(v.register( rule_name, self.hook_01_positive_integer_check))
        self.assertTrue(v.register( rule_name,
                         self.hook_01_positive_integer_check,
                         override = True))

        # rule must be as string
        self.assertFalse(v.register( 3.14, int))

        # rule must have a know type
        self.assertFalse(v.register( "pi", 3.14 ))

        # register an unallowed dict rule
        self.assertFalse(v.register( "pi", {} ))

    def test_validator_04_another_one(self):

        v = Validator()
        self.assertTrue(v.register( 'probability', self.hook_probability_check))

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
        self.assertTrue(v.register( rule_name, self.hook_02_positive_integer_check))

        # checks
        self.assertTrue( Validator().validate(1, rule_name))
        self.assertFalse( Validator().validate(-1, rule_name))
        self.assertFalse( Validator().validate(1.0, rule_name))


    def test_validator_06_strict_or_not(self):

        v = Validator()

        rule_name = 'this_rule_is_unknown'

        self.assertFalse( v.is_known_rule(rule_name) )
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

        v = Validator()

        training_args_ok = {
            'lr' : { 'rules': [ float, self.loss_rate_validation_hook] ,
                     'default': 1.0
                    },
        }

        self.assertTrue(v.register( "tr_01", training_args_ok))

        # and use it with it's name
        self.assertTrue( Validator().validate(
            { 'lr' : 0.4 } ,
            "tr_01"))

        # or directly
        self.assertTrue( Validator().validate(
            { 'lr' : 0.4 } ,
            training_args_ok ) )

        self.assertFalse( Validator().validate(
            { 'lr' : 'toto' } ,
            training_args_ok ) )

        self.assertFalse( Validator().validate(
            { 'lr' : 'toto' } ,
            training_args_ok ) )


        training_args_ko = {
            'lr' : { 'rules': [ self.loss_rate_validation_hook ] ,
                     'required': True,
                     'unallowed_key': False
                    },
        }

        # register the new rule
        self.assertFalse(v.register( "tr_02", training_args_ko))


    def test_validator_09_lambda(self):
        """
        check against a lambda expression
        """
        my_lambda = lambda a: isinstance(a, bool)

        v = Validator()
        self.assertTrue( v.validate( True, my_lambda) )
        self.assertTrue( v.validate( True, my_lambda) )
        self.assertFalse( v.validate( 3.14 , my_lambda) )


    def test_validator_10_validator_hook_type(self):
        """
        check _ValidatorHootType internal function
        """

        # invalid Type
        self.assertEqual( Validator()._hook_type( 3.14 ),
                         _ValidatorHookType.INVALID)

        self.assertEqual( Validator()._hook_type( None ),
                         _ValidatorHookType.INVALID)

        # builtin classes
        self.assertEqual( Validator()._hook_type( int ),
                         _ValidatorHookType.TYPECHECK)

        self.assertEqual( Validator()._hook_type( float ),
                         _ValidatorHookType.TYPECHECK)

        # non builtin classes
        self.assertEqual( Validator()._hook_type( Validator ),
                         _ValidatorHookType.TYPECHECK)

        # scheme validators
        v = SchemeValidator( {} )
        self.assertEqual( Validator()._hook_type( v ),
                         _ValidatorHookType.SCHEME_VALIDATOR)

        self.assertEqual( Validator()._hook_type( {} ),
                         _ValidatorHookType.SCHEME_AS_A_DICT)

        # functions
        self.assertEqual( Validator()._hook_type( self ),
                          _ValidatorHookType.FUNCTION)

        my_lambda = lambda : True
        self.assertEqual( Validator()._hook_type( my_lambda ),
                          _ValidatorHookType.LAMBDA)

    def test_validator_11_default_value(self):
        """
        check default field
        """

        # check that default value is conform to the rules
        sc = SchemeValidator( { "a": { "rules": [ str ],
                                       "default": "default_value"}
                               }
                             )
        self.assertTrue( sc.is_valid() )

        # this one is bad
        sc = SchemeValidator( { "a": { "rules": [ str ],
                                       "default": 1.0 }
                               }
                             )
        self.assertFalse( sc.is_valid() )


class TestSchemeValidator(unittest.TestCase):
    """
    unitests for SchemeValidator class
    """

    @staticmethod
    @validator_decorator
    def always_true_hook(value):
        return True

    def test_scheme_validator_01_validate_the_validator(self):
        """
        test SchemeValidator constructor
        """

        # empty scheme forbidden
        self.assertFalse( SchemeValidator( {} ) .is_valid())

        # but try to use it anyway
        v = SchemeValidator({})
        self.assertFalse( v.validate( {} ) )
        self.assertTrue( v.scheme() is None)

        # create abad validator, but stil use it
        v = SchemeValidator( { "toto": 1 } )
        self.assertFalse( v.is_valid())
        self.assertFalse( v.validate( {"toto": 1} ))

        # data should be properly defined
        self.assertFalse( SchemeValidator( { "data": [] } ) .is_valid())
        self.assertFalse( SchemeValidator( { "data": int } ) .is_valid())

        # grammar associated to data must be a non empty dict
        self.assertFalse( SchemeValidator( { "data": {} } ) .is_valid())

        # grammar associated to data must be a non empty dict containing
        # valid subkeys
        self.assertFalse( SchemeValidator( { "data": { "a": "b" } } ) .is_valid())

        # and the rules subkey must also be a non empty array
        self.assertFalse( SchemeValidator( { "data": { "rules": {}  } } ) .is_valid())
        self.assertFalse( SchemeValidator( { "data": { "rules": "b" } } ) .is_valid())
        self.assertFalse( SchemeValidator( { "data": { "rules": 1.0 } } ) .is_valid())
        # bad rules
        self.assertFalse( SchemeValidator( { "data": { "rules": [ 1.0 ] } } ) .is_valid())

        # empty array for rules is OK
        self.assertTrue( SchemeValidator( { "data": { "rules": [] } } ) .is_valid())

        # verify scheme() again
        grammar = { "data": { "rules": [] } }
        v = SchemeValidator( grammar )
        self.assertTrue( v.is_valid() )
        self.assertEqual( v.scheme(), grammar)
        self.assertFalse( v.validate( "not a dict" ) )
        self.assertFalse( v.validate( False ) )
        self.assertFalse( v.validate( None ) )
        self.assertFalse( v.validate( "data" ) )

        training_args_scheme = {
            'lr' : { 'rules': [ float, self.always_true_hook] ,
                     'default': 1.0
                    },
        }
        self.assertTrue( SchemeValidator( training_args_scheme ).is_valid())

        training_arg_scheme = {
            'lr' : { 'rules': [ self.always_true_hook],
                     'required': True,
                     'unallowed_key': False
                    },
        }
        self.assertTrue( SchemeValidator( training_args_scheme ).is_valid())


    def test_scheme_validator_02_validate_internal_hook_functions(self):
        """
        internal helpr function tests
        """

        #  check hook_type_validation
        self.assertTrue( Validator._is_hook_type_valid( float ))
        self.assertTrue( Validator._is_hook_type_valid( SchemeValidator ))
        self.assertTrue( Validator._is_hook_type_valid( {} ))
        self.assertTrue( Validator._is_hook_type_valid( self.always_true_hook ))

        self.assertFalse( Validator._is_hook_type_valid( 3.14 ))

        # check direct hook call
        self.assertTrue( Validator._hook_execute( 1.0, float ))
        status, error = Validator._hook_execute( "toto" , float )
        self.assertFalse(status)

        status, error = Validator._hook_execute( 1.0, "prout" )
        self.assertFalse(status)

        status, error = Validator._hook_execute( 1.0, SchemeValidator( {} ) )
        self.assertFalse(status)

        status, error = Validator._hook_execute( 1.0, {} )
        self.assertFalse(status)

    @staticmethod
    @validator_decorator
    def positive_integer(value):
        return isinstance(value, int) and value > 0

    def test_scheme_validator_03_validate_the_validator(self):
        """
        a more complicated scheme
        """

        training_args_scheme = {
            'lr' : { 'rules': [ float, lambda a: (a > 0) ],
                     'required': True,
                    },
            'round_limit' : { 'rules': [ self.positive_integer ],
                              'required': True,
                             },
            'batch_size' : { 'rules': [ self.positive_integer ],
                             'required': True,
                            },
            'epoch' : { 'rules': [ self.positive_integer ],
                        'required': True,
                       },
            'dry_run' : { 'rules': [ bool ],
                          'required': True,
                         },
            'batch_max_num' : { 'rules': [ self.positive_integer ],
                                'required': True,
                               },
        }
        v = SchemeValidator( training_args_scheme )
        self.assertTrue( v.is_valid() )

        training_args = {
            'batch_size': 20,
            'lr': 1e-5,
            'epochs': 1,
            'dry_run': False,
            'batch_maxnum': 250
        }
        self.assertFalse( v.validate( training_args ))

        training_args = {
            'batch_size': 20,
            'lr': 1e-5,
            'epochs': 1,
            'dry_run': False,
            'batch_maxnum': 250,
            'round_limit': 10
        }
        self.assertFalse( v.validate( training_args ))

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
