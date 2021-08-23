import unittest

import fedbiomed.common.message as message

class TestMessage(unittest.TestCase):
    '''
    Test the Message class
    '''
    # before the tests
    def setUp(self):
        pass

    # after the tests
    def tearDown(self):
        pass

    #
    # helper function to check failures for all Message classes
    # ---------------------------------------------------------
    def check_class_args(self,  cls, expected_result = True, **kwargs ):

        result = True

        # list of permitted classes
        all_classes = [
            message.SearchReply,
            message.PingReply,
            message.TrainReply,
            message.AddScalarReply,
            message.ErrorMessage,
            message.SearchRequest,
            message.PingRequest,
            message.TrainRequest
            ]

        # test minimal python (only affectation) to insure
        # that the exception will be trapped only on object affectation
        try:
            valid_class = False
            for c in all_classes:
                if cls == c:
                    #print("DEBUG: detected class:", c)
                    m = c(**kwargs)
                    valid_class = True
                    break

            # the tester passed a bad class name to check_class_args()
            if not valid_class:
                self.fail("check_class_args: bad class name")

        except:
            result = False

        # for DEBUG purpose
        #if expected_result is True:
        #    print("DEBUG [should be OK]:", kwargs)
        #else:
        #    print("DEBUG [should be KO]:", kwargs)

        # decode all cases
        if expected_result is True and result is True:
            self.assertTrue( True, "check_class_args: good params detected")

        if expected_result is True and result is False:
            self.fail( "check_class_args: good params detected as bad")

        if expected_result is False and result is True:
            self.fail( "check_class_args: bad params detected as good")

        if expected_result is False and result is False:
            self.assertTrue( True, "check_class_args: bad params correclty detected")

        pass

    def test_message(self):
        pass

    def test_searchreply(self):

        # verify necessary arguments of all message creation

        # well formatted message
        self.check_class_args(
            message.SearchReply,
            expected_result = True,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            client_id     = 'titi',
            command       = 'do_it')


        # all these test should fail (not enough arguments)
        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            count = 666 )

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            success = True)

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            databases = [1, 2, 3] )

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            client_id = 'toto')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            command = "toto" )

        # too much arguments
        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            client_id     = 'titi',
            command       = 'do_it',
            extra_arg     = "not_allowed"
        )

        # all the following should be bad (bad argument type)
        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = "not_an_integer",
            client_id     = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = True,
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            client_id     = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            client_id     = True,
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            client_id     = 'titi',
            command       = True)

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = "not a list",
            count         = 666,
            client_id     = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = "not_a_boolean",
            databases     = [],
            count         = 666,
            client_id     = 'titi',
            command       = 'do_it')

        pass

    def test_pingreply(self):

        # verify necessary arguments of all message creation

        # well formatted message
        self.check_class_args(
            message.PingReply,
            expected_result = True,

            researcher_id = 'toto',
            client_id     = 'titi',
            success       = True,
            command       = 'do_it')

        # bad formetted messages
        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            client_id = 'titi')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            success = False)

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            command = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id = 'titi',
            success = True,
            command = 'do_it',
            extra_arg = 'foobar')

        # bad argument type
        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = True,
            client_id     = 'titi',
            success       = True,
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = True,
            success       = True,
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            success       = 'not_a_bool',
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            success       = True,
            command       = True)

        pass


    def test_trainreply(self):

        # well formatted message
        self.check_class_args(
            message.TrainReply,
            expected_result = True,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        # bad param number
        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            job_id        = 'job')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            success       = True)

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            client_id     = 'titi')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            dataset_id    = 'my_data')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            params_url    = 'string_param')


        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            params_url    = 'string_param')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            timing        = { "t0": 0.0, "t1": 1.0})


        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            msg           = 'message_in_a_bottle')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it',
            extra_param   = 'dont_know_what_to_do_with_you')

        # bad param type
        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = True,
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = True,
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = 'not_a_bool',
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = True,
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = True,
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = True,
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = "not_a_dict",
            msg           = 'message_in_a_bottle',
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = True,
            command       = 'do_it')

        self.check_class_args(
            message.TrainReply,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job',
            success       = True,
            client_id     = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = True)

        pass


    def test_addScalarReply(self):
        # well formatted message

        self.check_class_args(
            message.AddScalarReply,
            expected_result = True,

            researcher_id = 'toto',
            client_id     = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')


        # bad param number
        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto')


        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            client_id     = 'titi')


        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            job_id        = 'tutu')


        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            key           = 3.14)


        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            iteration     = 666)


        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            command       = 'do_it')


        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it',
            extra_arg     = '???')



        # bad param type
        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = False,
            client_id     = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = False,
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            job_id        = False,
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            job_id        = 'tutu',
            key           = "not_a_float",
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = "no_an_int",
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            client_id     = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = False)


        pass


    def test_errormessage(self):

        # well formatted message
        self.check_class_args(
            message.ErrorMessage,
            expected_result = True,

            researcher_id = 'toto',
            success       = True,
            client_id     = 'titi',
            msg           = 'this is an error message',
            command       = 'do_it')


        # bad param number
        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            success       = True)

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            client_id     = 'titi')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            msg           = 'this is an error message')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            command       = 'do_it')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            client_id     = 'titi',
            msg           = 'this is an error message',
            command       = 'do_it',
            extra_arg     = '???' )


        # bad param type
        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = False,
            success       = True,
            client_id     = 'titi',
            msg           = 'this is an error message',
            command       = 'do_it')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            success       = "not_a_bool",
            client_id     = 'titi',
            msg           = 'this is an error message',
            command       = 'do_it')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            client_id     = False,
            msg           = 'this is an error message',
            command       = 'do_it')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            client_id     = 'titi',
            msg           = [ 1 , 2 ],
            command       = 'do_it')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            client_id     = 'titi',
            msg           = 'this is an error message',
            command       = False)


        pass


    def test_searchrequest(self):
        # well formatted message
        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            researcher_id = 'toto')


        # bad param number
        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            tags          = [ "data", "doto" ])

        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            command       = 'do_it')

        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            researcher_id = 'toto',
            tags          = [ "data", "doto" ],
            command       = 'do_it',
            extra_args    = '???' )


        # bad param type
        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            researcher_id = False,
            tags          = [ "data", "doto" ],
            command       = 'do_it')

        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            researcher_id = 'toto',
            tags          = "not_a_list",
            command       = 'do_it')

        self.check_class_args(
            message.SearchRequest,
            expected_result = False,

            researcher_id = 'toto',
            tags          = [ "data", "doto" ],
            command       = False)


        pass


    def test_pingrequest(self):
        # well formatted message
        self.check_class_args(
            message.PingRequest,
            expected_result = True,

            researcher_id = 'toto',
            command       = 'do_it')



        # bad param number
        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            command       = 'do_it')

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto',
            command       = 'do_it',
            extra_arg     = '???')


        # bad param type
        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = False,
            command       = 'do_it')

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto',
            command       = False)


        pass


    def test_trainrequest(self):
        # well formatted message
        self.check_class_args(
            message.TrainRequest,
            expected_result = True,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')



        # bad param number
        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            job_id        = 'job_number')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            params_url    = 'this_is_an_url')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            training_args = { "a": 1, "b": 2})

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            training_data = { "data" : "MNIS"})

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            model_args    = { "c": 3, "d": 4})

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            model_url     = "http://dev.null")

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            model_class   = 'my_model')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it',
            extra_arg     = '???')


        # bad param type
        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = False,
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = False,
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = False,
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = "not_a_dict",
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = "not_a_dict",
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = "not_a_dict",
            model_url     = "http://dev.null",
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = False,
            model_class   = 'my_model',
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = False,
            command       = 'do_it')

        self.check_class_args(
            message.TrainRequest,
            expected_result = False,

            researcher_id = 'toto',
            job_id        = 'job_number',
            params_url    = 'this_is_an_url',
            training_args = { "a": 1, "b": 2},
            training_data = { "data" : "MNIS"},
            model_args    = { "c": 3, "d": 4},
            model_url     = "http://dev.null",
            model_class   = "my_model",
            command       = False)


        pass


    def test_researchermessages(self):
        # well formatted message

        # bad param number

        # bad param type

        pass


    def test_nodemessages(self):
        # well formatted message

        # bad param number

        # bad parma type

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
