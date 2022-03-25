import unittest

import fedbiomed.common.json as js
from fedbiomed.common.constants import ErrorNumbers


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

    def test_basic_json(self):
        '''
        serialized/deserialize test
        '''
        msg = '{"foo": "bar"}'
        self.assertEqual(js.deserialize_msg(msg)["foo"], "bar")

        loop1 = js.serialize_msg(js.deserialize_msg(msg))
        self.assertEqual(loop1, msg)

        pass

    def test_errnum_json(self):
        '''
        serialized/deserialize errnum
        '''

        for e in ErrorNumbers:
            dict = {
                'errnum': e
            }

            dict2json = js.serialize_msg(dict)
            self.assertTrue(str(e.value[0]) in dict2json)

            json2dict = js.deserialize_msg(dict2json)
            self.assertEqual(e, json2dict['errnum'])

        # test also unknow error
        serial = '{"errnum": 123456789}'
        json2dict = js.deserialize_msg(serial)
        self.assertEqual(json2dict['errnum'], ErrorNumbers.FB999)

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
