import unittest
from dataclasses import dataclass

import fedbiomed.common.message as message
from fedbiomed.common.constants import ErrorNumbers

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
            message.LogMessage,
            message.ErrorMessage,
            message.SearchRequest,
            message.PingRequest,
            message.TrainRequest,
            message.ListReply,
            message.ListRequest,
            message.ModelStatusReply,
            message.ModelStatusRequest
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

    @dataclass
    class dummyMessage(message.Message):
        """
        dummy class to fully test the Message class
        """
        a: int
        b: str


    def test_message(self):

        m1 = message.Message()

        # initial dictionnary is empty
        self.assertEqual( m1.get_dict(), {} )

        # get/set tester
        m1.set_param( "a", 1);
        self.assertEqual( m1.get_dict(), { "a" : 1} )
        self.assertEqual( m1.get_param( "a") , 1)

        #
        m1.set_param( "a", 2);
        m1.set_param( "b", "this_is_a_string");
        self.assertEqual( m1.get_param( "a") , 2)
        self.assertEqual( m1.get_param( "b") , "this_is_a_string")

        # this constructor is not validated until validate() is
        # effectively called
        m2 = self.dummyMessage( a = 1 , b = "oh my god !")
        self.assertEqual( m2.get_param( "a") , 1)
        self.assertEqual( m2.get_param( "b") , "oh my god !")

        # too difficult to test validate directly
        # it is indirectly tested by the other test_*() calls

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
            node_id       = 'titi',
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

            node_id = 'toto')

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
            node_id       = 'titi',
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
            node_id       = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = True,
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = True,
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = 'titi',
            command       = True)

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = "not a list",
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.SearchReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = "not_a_boolean",
            databases     = [],
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')

        pass

    def test_pingreply(self):

        # verify necessary arguments of all message creation

        # well formatted message
        self.check_class_args(
            message.PingReply,
            expected_result = True,

            researcher_id = 'toto',
            node_id       = 'titi',
            sequence      = 100,
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

            node_id   = 'titi')

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

            sequence = 100
            )

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id = 'titi',
            success = True,
            sequence = 100,
            command = 'do_it',
            extra_arg = 'foobar')

        # bad argument type
        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = True,
            node_id       = 'titi',
            success       = True,
            sequence = 100,
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = True,
            success       = True,
            sequence = 100,
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            success       = 'not_a_bool',
            sequence = 100,
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            success       = 'not_a_bool',
            sequence      = False,
            command       = 'do_it')

        self.check_class_args(
            message.PingReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
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
            node_id       = 'titi',
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

            node_id       = 'titi')

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
            node_id       = 'titi',
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
            node_id       = 'titi',
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
            node_id       = 'titi',
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
            node_id       = 'titi',
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
            node_id       = True,
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
            node_id       = 'titi',
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
            node_id       = 'titi',
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
            node_id       = 'titi',
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
            node_id       = 'titi',
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
            node_id       = 'titi',
            dataset_id    = 'my_data',
            params_url    = 'string_param',
            timing        = { "t0": 0.0, "t1": 1.0},
            msg           = 'message_in_a_bottle',
            command       = True)

        pass

    def test_listreply(self):

        # well formatted message
        self.check_class_args(
            message.ListReply,
            expected_result = True,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')


        # all these test should fail (not enough arguments)
        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            count = 666 )

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            success = True)

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            databases = [1, 2, 3] )

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            node_id = 'toto')

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            command = "toto" )

        # too much arguments
        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it',
            extra_arg     = "not_allowed"
        )

        # all the following should be bad (bad argument type)
        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = "not_an_integer",
            node_id       = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = True,
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = True,
            command       = 'do_it')

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = [1, 2, 3],
            count         = 666,
            node_id       = 'titi',
            command       = True)

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = True,
            databases     = "not a list",
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')

        self.check_class_args(
            message.ListReply,
            expected_result = False,

            researcher_id = 'toto',
            success       = "not_a_boolean",
            databases     = [],
            count         = 666,
            node_id       = 'titi',
            command       = 'do_it')



    def test_addscalarreply(self):
        # well formatted message

        self.check_class_args(
            message.AddScalarReply,
            expected_result = True,

            researcher_id = 'toto',
            node_id       = 'titi',
            job_id        = 'tutu',
            key           = 'key',
            value         = 13.34,
            epoch         = 12,
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

            node_id       = 'titi')


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
            node_id       = 'titi',
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
            node_id       = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = False,
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            job_id        = False,
            key           = 3.14,
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            job_id        = 'tutu',
            key           = "not_a_float",
            iteration     = 666,
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = "no_an_int",
            command       = 'do_it')

        self.check_class_args(
            message.AddScalarReply,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            job_id        = 'tutu',
            key           = 3.14,
            iteration     = 666,
            command       = False)


        pass

    def test_modelstatusreply(self):

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = True,

            researcher_id           = 'toto',
            node_id                 = 'titi',
            job_id                  = 'titi',
            success                 = True,
            approval_obligation     = True,
            is_approved             = True,
            msg                     =  'sdrt',
            model_url               = 'url',
            command                 = 'do_it')

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            researcher_id           = 'toto',
            node_id                 = 12334,
            job_id                  = 'titi',
            success                 = True,
            approval_obligation     = True,
            is_approved             = True,
            msg                     = 'sdrt',
            model_url               = 'url',
            command                 = 'do_it')

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            researcher_id           = 12344,
            node_id                 = '12334',
            job_id                  = 'titi',
            success                 = True,
            approval_obligation     = True,
            is_approved             = True,
            msg                     =  'sdrt',
            model_url               = 'url',
            command                 = 'do_it')

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            researcher_id           = '12344',
            node_id                 = '12334',
            job_id                  = 'titi',
            success                 = True,
            approval_obligation     = True,
            is_approved             = 'True',
            msg                     =  'sdrt',
            model_url               = 'url',
            command                 = 'do_it')

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            researcher_id           = '12344',
            node_id                 = '12334',
            job_id                  = 'titi',
            success                 = True,
            approval_obligation     = 'True',
            is_approved             =  True,
            msg                     =  'sdrt',
            model_url               = 'url',
            command                 = 'do_it')

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            researcher_id           = 333,
            node_id                 = 1212,
            job_id                  = False,
            success                 = 'not a bool',
            approval_obligation     = True,
            is_approved             = True,
            msg                     =  'sdrt',
            model_url               = 123123,
            command                 = True)

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            researcher_id           = 333,
            node_id                 = 1212,
            job_id                  = False,
            success                 = 'not a bool',
            approval_obligation     = True,
            is_approved             = True,
            msg                     =  'sdrt')

        self.check_class_args(
            message.ModelStatusReply,
            expected_result = False,

            success                 = 'not a bool',
            approval_obligation     = True,
            is_approved             = True,
            msg                     =  'sdrt')



    def test_logmessage(self):

        # well formatted message
        self.check_class_args(
            message.LogMessage,
            expected_result = True,

            researcher_id = 'toto',
            node_id       = 'titi',
            level         = 'INFO',
            msg           = 'this is an error message',
            command       = 'log'
        )


        # bad param number
        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            node_id       = 'titi')

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            level        = 'INFO')

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            msg           = 'this is an error message')

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            command       = 'log')

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            level         = 'INFO',
            msg           = 'this is an error message',
            command       = 'log',
            extra_arg     = '???' )


        # bad param type
        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = False,
            node_id       = 'titi',
            level         = 'INFO',
            msg           = 'this is an error message',
            command       = 'log'
        )

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = False,
            level         = 'INFO',
            msg           = 'this is an error message',
            command       = 'log'
        )

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            level         = False,
            msg           = 'this is an error message',
            command       = 'log'
        )

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            level         = 'INFO',
            msg           = [ 1 , 2 ],
            command       = 'log')

        self.check_class_args(
            message.LogMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            level         = 'INFO',
            msg           = [ 1 , 2 ],
            command       = False)


        pass

    def test_errormessage(self):

        # well formatted message
        self.check_class_args(
            message.ErrorMessage,
            expected_result = True,

            researcher_id = 'toto',
            node_id       = 'titi',
            errnum        = ErrorNumbers.FB100,
            msg           = 'this is an error message',
            command       = 'log'
        )


        # bad param number
        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            node_id       = 'titi')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            errnum       = ErrorNumbers.FB100)

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            msg           = 'this is an error message')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            command       = 'log')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            errnum        = ErrorNumbers.FB100,
            msg           = 'this is an error message',
            command       = 'log',
            extra_arg     = '???' )


        # bad param type
        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = False,
            node_id       = 'titi',
            errnum        = ErrorNumbers.FB100,
            msg           = 'this is an error message',
            command       = 'log'
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = False,
            errnum        = ErrorNumbers.FB100,
            msg           = 'this is an error message',
            command       = 'log'
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            errnum        = False,
            msg           = 'this is an error message',
            command       = 'log'
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            errnum        = ErrorNumbers.FB100,
            msg           = [ 1 , 2 ],
            command       = 'log')

        self.check_class_args(
            message.ErrorMessage,
            expected_result = False,

            researcher_id = 'toto',
            node_id       = 'titi',
            errnum        = ErrorNumbers.FB100,
            msg           = [ 1 , 2 ],
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
            sequence       = 100,
            command       = 'do_it')



        # bad param number
        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto')

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            sequence       = 100)

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            command       = 'do_it')

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto',
            command       = 'do_it',
            sequence       = 100,
            extra_arg     = '???')


        # bad param type
        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = False,
            sequence       = 100,
            command       = 'do_it')

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto',
            sequence       = False,
            command       = False)

        self.check_class_args(
            message.PingRequest,
            expected_result = False,

            researcher_id = 'toto',
            sequence       = 100,
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
    def test_listrequest(self):

        # well formatted message
        self.check_class_args(
            message.ListRequest,
            expected_result = True,
            researcher_id='toto',
            command='sada')

        # bad param number
        self.check_class_args(
            message.ListRequest,
            expected_result = False,
            tags          = [ "data", "doto" ])

        self.check_class_args(
            message.ListRequest,
            expected_result = False,
            command       = 'do_it')

        self.check_class_args(
            message.ListRequest,
            expected_result = False,
            researcher_id = 'toto',
            tags          = [ "data", "doto" ],
            command       = 'do_it',
            extra_args    = '???' )


        # bad param type
        self.check_class_args(
            message.ListRequest,
            expected_result = False,
            researcher_id = False,
            tags          = [ "data", "doto" ],
            command       = 'do_it')

        # bad param type
        self.check_class_args(
            message.ListRequest,
            expected_result = False,
            researcher_id = False,
            command       = True)

        pass


    def test_modelstatusrequest(self):

        self.check_class_args(
            message.ModelStatusRequest,
            expected_result = True,

            researcher_id   = 'toto',
            job_id          = 'sdsd',
            model_url       = 'do_it',
            command         = 'command-dummy' )


        self.check_class_args(
            message.ModelStatusRequest,
            expected_result = False,

            researcher_id   = True,
            job_id          = 'sdsd',
            model_url       = 'do_it',
            command         = 'command-dummy' )

        self.check_class_args(
            message.ModelStatusRequest,
            expected_result = False,

            researcher_id   = 'toto',
            job_id          = 122323,
            model_url       = 'do_it',
            command         = 'command-dummy' )

        self.check_class_args(
            message.ModelStatusRequest,
            expected_result = False,

            researcher_id   = 'toto',
            job_id          = 'sdsd',
            model_url       = 12323,
            command         = 'command-dummy' )

        self.check_class_args(
            message.ModelStatusRequest,
            expected_result = False,

            researcher_id   = 'ttot',
            job_id          = 'sdsd',
            model_url       = 'do_it',
            command         = False )

    # test ResearcherMessage and NodeMessagess classes
    # (next 9 tests)
    def test_trainmessages(self):

        params = {
            "researcher_id" : 'toto',
            "job_id"        : 'job',
            "success"       : True,
            "node_id"       : 'titi',
            "dataset_id"    : 'my_data',
            "params_url"    : 'string_param',
            "timing"        : { "t0": 0.0, "t1": 1.0},
            "msg"           : 'message_in_a_bottle',
            "command"       : 'train' }

        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.TrainReply )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.TrainReply )

        params = {
            "researcher_id" : 'toto',
            "job_id"        : 'job',
            "params_url"    : "https://dev.null",
            "training_args" : { } ,
            "training_data" : { } ,
            "model_args"    : { } ,
            "model_url"     : "https://dev.null",
            "model_class"   : "my_model",
            "command"       : 'train' }

        r = message.ResearcherMessages.request_create( params )
        self.assertIsInstance( r, message.TrainRequest )

        r = message.NodeMessages.request_create( params )
        self.assertIsInstance( r, message.TrainRequest )

    def test_listmessages(self):

        """  Test list datasets messages for node and researcher """
        params = {
            "researcher_id" : 'toto',
            "success"       : True,
            "databases"     : [ "one", "two" ],
            "count"         : 666,
            "node_id"       : 'titi',
            "command"       : 'list' }

        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.ListReply )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.ListReply )


        params = {
            "researcher_id" : 'toto',
            "command"       : 'list' }
        r = message.ResearcherMessages.request_create( params )
        self.assertIsInstance( r, message.ListRequest )

        r = message.NodeMessages.request_create( params )
        self.assertIsInstance( r, message.ListRequest )


    def test_searchmessages(self):

        params = {
            "researcher_id" : 'toto',
            "success"       : True,
            "databases"     : [ "one", "two" ],
            "count"         : 666,
            "node_id"       : 'titi',
            "command"       : 'search' }

        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.SearchReply )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.SearchReply )


        params = {
            "researcher_id" : 'toto',
            "tags"          : [],
            "command"       : 'search' }
        r = message.ResearcherMessages.request_create( params )
        self.assertIsInstance( r, message.SearchRequest )

        r = message.NodeMessages.request_create( params )
        self.assertIsInstance( r, message.SearchRequest )


    def test_pingmessages(self):

        # ping
        params = {
            "researcher_id" : 'toto' ,
            "node_id"       : 'titi' ,
            "sequence"      : 100,
            "success"       : True,
            "command"       : 'pong'
        }
        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.PingReply )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.PingReply )

        params = {
            "researcher_id" : 'toto' ,
            "sequence"      : 100,
            "command"       : 'ping'
        }
        r = message.ResearcherMessages.request_create( params )
        self.assertIsInstance( r, message.PingRequest )

        r = message.NodeMessages.request_create( params )
        self.assertIsInstance( r, message.PingRequest )


    def test_logmessages(self):

        # error
        params = {
            "researcher_id" : 'toto' ,
            "node_id"       : 'titi' ,
            "level"         : 'INFO',
            "msg"           : 'bim boum badaboum',
            "command"       : 'log'
        }
        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.LogMessage )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.LogMessage )

    def test_errormessages(self):

        # error
        params = {
            "researcher_id" : 'toto' ,
            "node_id"       : 'titi' ,
            "errnum"        : ErrorNumbers.FB100,
            "msg"           : 'bim boum badaboum',
            "command"       : 'error'
        }
        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.ErrorMessage )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.ErrorMessage )

    def test_addscalarmessages(self):

        # addScalar
        params = {
            "researcher_id" : 'toto' ,
            "node_id"       : 'titi' ,
            "job_id"        : 'job_id',
            "key"           : 'key',
            "value"         : 14.34,
            "iteration"     : 666,
            "epoch"         : 12,
            "command"       : 'add_scalar'
        }
        r = message.ResearcherMessages.reply_create( params )
        self.assertIsInstance( r, message.AddScalarReply )

        r = message.NodeMessages.reply_create( params )
        self.assertIsInstance( r, message.AddScalarReply )

    def test_unknowmessages(self):
        # we only test one error (to get 100% coverage)
        # all test have been made above

        params = { 'command' : 'unknown'}

        try:
            r = message.ResearcherMessages.reply_create( params )
            # should not reach this line
            self.fail("unknown reply message type for researcher not detected")

        except:
            # should be reached
            self.assertTrue( True, "unknown reply message type for researcher detected")

        try:
            r = message.ResearcherMessages.request_create( params )
            # should not reach this line
            self.fail("unknown request message type for researcher not detected")

        except:
            # should be reached
            self.assertTrue( True, "unknown request message type for researcher detected")
        pass

        try:
            r = message.NodeMessages.reply_create( params )
            # should not reach this line
            self.fail("unknown reply message type for node not detected")

        except:
            # should be reached
            self.assertTrue( True, "unknown reply message type for node detected")

        try:
            r = message.NodeMessages.request_create( params )
            # should not reach this line
            self.fail("unknown request message type for node not detected")

        except:
            # should be reached
            self.assertTrue( True, "unknown request message type for node detected")
        pass

    def test_model_status_messages(self):

        params_reply =  {
            'researcher_id'           : 'toto',
            'node_id'                 : 'titi',
            'job_id'                  : 'titi',
            'success'                 : True,
            'approval_obligation'     : True,
            'is_approved'             : True,
            'msg'                     :  'sdrt',
            'model_url'               : 'url',
            'command'                 : 'model-status'
        }

        r = message.ResearcherMessages.reply_create( params_reply )
        self.assertIsInstance( r, message.ModelStatusReply )

        r = message.NodeMessages.reply_create( params_reply )
        self.assertIsInstance( r, message.ModelStatusReply )

        params_request = {
            'researcher_id'   : 'toto',
            "job_id"          : 'titi',
            "model_url"       : 'url-dummy',
            "command"         : 'model-status'
            }

        r = message.ResearcherMessages.request_create( params_request )
        self.assertIsInstance( r, message.ModelStatusRequest )

        r = message.NodeMessages.request_create( params_request )
        self.assertIsInstance( r, message.ModelStatusRequest )

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
