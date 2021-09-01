import unittest

import fedbiomed.common.json as js

class TestJson(unittest.TestCase):
    '''
    Test the Json class
    '''
    # before the tests
    def setUp(self):
        pass

    # after the tests
    def tearDown(self):
        pass


    def test_json(self):
        '''
        still do not understand why we use this wrapper.....
        '''
        msg = '{"foo": "bar"}'
        self.assertEqual( js.deserialize_msg(msg)["foo"] , "bar")

        loop1 = js.serialize_msg(js.deserialize_msg(msg))
        self.assertEqual( loop1, msg )

        loop2 = js.deserialize_msg(js.serialize_msg(msg))
        self.assertEqual( loop1, msg )
        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
