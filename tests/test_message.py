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

    # helper function to check failures
    def check_searchreply_args(self,  expected_result = True, **kwargs ):

        result = True
        try:
            # test minimal python, to check exactly what we want
            m = message.SearchReply(**kwargs)

        except:
            result = False

        #
        if expected_result is True:
            print("DEBUG [should be OK]:", kwargs)
        else:
            print("DEBUG [should be KO]:", kwargs)

        # decode all cases
        if expected_result is True and result is True:
            self.assertTrue( True, "SearchReply good params detected")

        if expected_result is True and result is False:
            self.fail( "SearchReply (good) params detected as bad")

        if expected_result is False and result is True:
            self.fail( "SearchReply (bad) params detected as good")

        if expected_result is False and result is False:
            self.assertTrue( True, "SearchReply bad params correclty detected")


    def test_message(self):
        pass

    def test_searchreply(self):

        # verify necessary arguments
        # all these test should fail (not enough arguments)
        self.check_searchreply_args( expected_result = False, researcher_id = 'toto')
        self.check_searchreply_args( expected_result = False, count = 666 )
        self.check_searchreply_args( expected_result = False, success = True)
        self.check_searchreply_args( expected_result = False, databases = [1, 2, 3] )
        self.check_searchreply_args( expected_result = False, client_id = 'toto')
        self.check_searchreply_args( expected_result = False, command = "toto" )

        # this one should be good
        self.check_searchreply_args( expected_result = True,
            researcher_id = 'toto',
            success = True,
            databases = [1, 2, 3],
            count = 666,
            client_id = 'titi',
            command = 'do_it')

        # the following should be bad (bad argument type)
        self.check_searchreply_args( expected_result = False,
            researcher_id = 'toto',
            success = True,
            databases = [1, 2, 3],
            count = "not_an_integer",
            client_id = 'titi',
            command = 'do_it')



    def test_pingreply(self):
        pass


    def test_trainreply(self):
        pass


    def test_addScalarreply(self):
        pass


    def test_errormessage(self):
        pass


    def test_searchrequest(self):
        pass


    def test_pingrequest(self):
        pass


    def test_trainrequest(self):
        pass


    def test_researchermessages(self):
        pass


    def test_nodemessages(self):
        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
