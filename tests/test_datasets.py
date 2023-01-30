import unittest

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from fedbiomed.researcher.datasets import FederatedDataSet


class TestFederatedDataset(ResearcherTestCase):
    """
    Test `FederatedDataset` class
    Args:
        unittest ([type]): [description]
    """

    # before the tests
    def setUp(self):
        self.data = {
            'node-1': [{'dataset_id': 'dataset-id-1',
                        'shape': [100, 100]}],
            'node-2': [{'dataset_id': 'dataset-id-2',
                        'shape': [120, 120], 
                        'test_ratio': .0}],
        }

        self.fds = FederatedDataSet(self.data)

    # after the tests
    def tearDown(self):
        pass

    def test_federated_dataset_02_data(self):
        """ Testing property .data()
        """

        updated_data = self.fds.data()
        # federated dataset should have added a new entry `test_ratio` in the FederatedDataset
        self.assertDictEqual(self.data, updated_data, 'Can not get data properly from FederatedDataset')

    def test_federated_dataset_04_node_ids(self):
        """ Testing node_ids getter/properties
            FIXME: When refactoring properties as getters
        """

        node_ids = self.fds.node_ids()
        self.assertListEqual(node_ids, ['node-1', 'node-2'], 'Can not get node ids of FederatedDataset properly')

    def test_federated_dataset_05_sample_sizes(self):
        """ Testing node_ids getter/properties
           FIXME: When refactoring properties as getters
       """
        # Nothing to do it is an empty method
        sizes = [val[0]["shape"][0] for (_, val) in self.data.items()]
        sample_sizes = self.fds.sample_sizes()
        self.assertListEqual(sizes, sample_sizes, 'Provided sample sizes and result of sample_sizes do not match')

    def test_federated_dataset_06_shapes(self):
        """ Testing shapes property of FederatedDataset """

        node_1 = list(self.data.keys())[0]
        node_2 = list(self.data.keys())[1]

        size_1 = self.data[node_1][0]['shape'][0]
        size_2 = self.data[node_2][0]['shape'][0]

        shapes = self.fds.shapes()
        self.assertEqual(shapes[node_1], size_1)
        self.assertEqual(shapes[node_2], size_2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
