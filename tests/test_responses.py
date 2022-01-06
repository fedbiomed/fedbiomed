# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import unittest

from fedbiomed.researcher.responses import Responses

class TestResponses(unittest.TestCase):
    '''
    Test the Responses class
    '''
    # before the tests
    def setUp(self):
        self.r1 = Responses( [] )
        self.r2 = Responses( [ 'foo' ] )
        self.r3 = Responses( { 'key' : 'value' }  )

    # after the tests
    def tearDown(self):
        pass


    # tests
    def test_basic(self):
        self.assertEqual( len(self.r1.data) , 0 )

        # test __len__
        self.assertEqual( len(self.r1) , 0 )

        # test __getitem__
        self.assertEqual( self.r3[0], {"key" : "value"} )

    def test_append(self):
        self.assertEqual( len(self.r3) , 1 )

        self.r3.append(self.r3)
        self.assertEqual( len(self.r3) , 2 )

        self.r3.append(self.r3)
        self.assertEqual( len(self.r3) , 4 )

        self.assertEqual( self.r3[0], {"key" : "value"} )
        self.assertEqual( self.r3[1], {"key" : "value"} )
        self.assertEqual( self.r3[2], {"key" : "value"} )
        self.assertEqual( self.r3[3], {"key" : "value"} )

        self.r3.append( [ "something" ])
        self.assertEqual( len(self.r3) , 5 )

        self.r3.append( { "foo" : "bar" } )
        self.assertEqual( len(self.r3) , 6 )


    def test_str(self):
        string = str(self.r3)
        self.assertIsInstance(string, str)

    def test_dataframe(self):
        df = self.r1.dataframe
        self.assertEqual( len(df) , 0 )

        df = self.r2.dataframe
        self.assertEqual( len(df) , 1 )

        df = self.r3.dataframe
        self.assertEqual( len(df) , 1 )

    def test_setter_getter(self):
        r = Responses( [] )
        self.assertEqual( len(r) , 0 )

        # set_data dict
        data1 = { "titi" : "toto" }
        r1 = Responses( [] )
        self.assertEqual( len(r1) , 0)
        r1.set_data( data1 )
        self.assertEqual( len(r1) , len(data1) )
        self.assertEqual( r1.get_data() , data1 )

        # set_data lis
        data2 = [ "titi", "toto" ]
        r2 = Responses( [] )
        self.assertEqual( len(r2) , 0)
        r2.set_data( data2 )
        self.assertEqual( len(r2) , len(data2) )
        self.assertEqual( r2.get_data() , data2 )


if __name__ == '__main__': # pragma: no cover
    unittest.main()
