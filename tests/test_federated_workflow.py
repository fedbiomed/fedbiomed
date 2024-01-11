import unittest
from unittest.mock import patch

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
#############################################################

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows import FederatedWorkflow


class TestFederatedWorkflow(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests")
        #super().setUp()
        self.abstract_methods_patcher = patch.multiple(FederatedWorkflow, __abstractmethods__=set())
        self.abstract_methods_patcher.start()

    def tearDown(self):
        super().tearDown()
        self.abstract_methods_patcher.stop()

    def test_federated_workflow_01_initialization(self):
        # FederatedWorkflow must be default-constructible
        exp = FederatedWorkflow()
        # FederatedWorkflow has a lot of setters
        # tags
        exp.set_tags('just-a-str')
        self.assertEqual(exp.tags(), ['just-a-str'])
        exp.set_tags(None)
        self.assertIsNone(exp.tags())
        exp.set_tags(['first', 'second'])
        self.assertEqual(exp.tags(), ['first', 'second'])
        # nodes
        exp.set_nodes(None)
        self.assertIsNone(exp.nodes())
        exp.set_nodes(['first', 'second'])
        self.assertEqual(exp.nodes(), ['first', 'second'])
        # training_data
        exp.set_training_data(None, from_tags=False)
        self.assertIsNone(exp.training_data())
        self.fake_search_reply = {'node1': [{'my-metadata': 'is-the-best'}]}
        self.mock_requests.return_value.search.return_value = self.fake_search_reply
        exp.set_training_data(None, from_tags=True)
        self.assertDictEqual(exp.training_data().data(), {'node1': {'my-metadata': 'is-the-best'}})







if __name__ == '__main__':  # pragma: no cover
    unittest.main()
