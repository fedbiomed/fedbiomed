import unittest

from fedbiomed.researcher.responses import Responses

class TestResponses(unittest.TestCase):
    '''
    Test the Responses class
    '''
    # before the tests
    def setUp(self):
        self.r1 = Responses( [] )
        self.r2 = Responses( [] )
        self.r3 = Responses( { 'key' : 'value' }  )

    # after the tests
    def tearDown(self):
        pass


    # tests
    def test_basic(self):
        self.assertEqual( len(self.r1.data) , 0 )

        # test __len__
        self.assertEqual( len(self.r1) , 0 )

    def test_concat(self):
        self.assertEqual( len(self.r3) , 1 )

        self.r3.append(self.r3)
        self.assertEqual( len(self.r3) , 2 )

        self.r3.append(self.r3)
        self.assertEqual( len(self.r3) , 4 )

        self.r3.append( [ "something" ])
        self.assertEqual( len(self.r3) , 5 )


    def test_str(self):
        string = str(self.r3)
        self.assertIsInstance(string, str)

if __name__ == '__main__':
    unittest.main()
