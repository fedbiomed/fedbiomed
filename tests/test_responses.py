import unittest

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from fedbiomed.researcher.responses import Responses
from fedbiomed.common.exceptions import FedbiomedResponsesError


class TestResponses(ResearcherTestCase):
    '''
    Test the Responses class
    '''

    # before the tests
    def setUp(self):
        self.r1 = Responses([])
        self.r2 = Responses(['foo'])
        self.r3 = Responses({'key': 'value'})

    # after the tests
    def tearDown(self):
        pass

    # tests
    def test_responses_01_basic(self):
        self.assertEqual(len(self.r1.data()), 0)

        # test __len__
        self.assertEqual(len(self.r1), 0)

        # test __getitem__
        self.assertEqual(self.r3[0], {"key": "value"})

    def test_responses_02_append(self):
        self.assertEqual(len(self.r3), 1)

        self.r3.append(self.r3)
        self.assertEqual(len(self.r3), 2)

        self.r3.append(self.r3)
        self.assertEqual(len(self.r3), 4)

        self.assertEqual(self.r3[0], {"key": "value"})
        self.assertEqual(self.r3[1], {"key": "value"})
        self.assertEqual(self.r3[2], {"key": "value"})
        self.assertEqual(self.r3[3], {"key": "value"})

        self.r3.append(["something"])
        self.assertEqual(len(self.r3), 5)

        self.r3.append({"foo": "bar"})
        self.assertEqual(len(self.r3), 6)

        with self.assertRaises(FedbiomedResponsesError):
            self.r3.append('STRING')

    def test_responses_03_magic_str(self):
        string = str(self.r3)
        self.assertIsInstance(string, str)

    def test_responses_04_dataframe(self):
        df = self.r1.dataframe()
        self.assertEqual(len(df), 0)

        df = self.r2.dataframe()
        self.assertEqual(len(df), 1)

        df = self.r3.dataframe()
        self.assertEqual(len(df), 1)

    def test_responses_05_setter_getter(self):
        r = Responses([])
        self.assertEqual(len(r), 0)

        # set_data dict
        data1 = [{"titi": "toto"}]
        r1 = Responses([])
        self.assertEqual(len(r1), 0)
        r1.set_data(data1)
        self.assertEqual(len(r1), len(data1))
        self.assertListEqual(r1.data(), data1)

        # set_data lis
        data2 = ["titi", "toto"]
        r2 = Responses([])
        self.assertEqual(len(r2), 0)
        r2.set_data(data2)
        self.assertEqual(len(r2), len(data2))
        self.assertListEqual(r2.data(), data2)

        with self.assertRaises(FedbiomedResponsesError):
            data1 = {"titi": "toto"}
            r1 = Responses([])
            r1.set_data(data1)
            
    def test_response_06_get_index_from_node_id(self):
        r = Responses([])
        node_1_resp = {'node_id': 'node_1', 'params': [1, 2, 4]}
        r.append(node_1_resp)
        
        self.assertEqual(r.get_index_from_node_id('node_1'), 0)

        node_2_resp = {'node_id': 'node_2', 'value': 3}
        r.append(node_2_resp)
        r.append(['foo'])
        self.assertEqual(r.get_index_from_node_id('node_1'), 0)
        self.assertEqual(r.get_index_from_node_id('node_2'), 1)
        self.assertIsNone(r.get_index_from_node_id('node_x'))
        
        r.append(r)
        self.assertEqual(r.get_index_from_node_id('node_1'), 0)
        self.assertEqual(r.get_index_from_node_id('node_2'), 1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
