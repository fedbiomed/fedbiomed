import unittest

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
#############################################################

from fedbiomed.researcher.federated_workflows import Experiment


class TestExperiment(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests")

    def tearDown(self):
        super().tearDown()

    def test_experiment_01_initialization(self):
        # Experiment must be default-constructible
        exp = Experiment()


        # Experiment has a lot of setters
        exp.set_tags()




if __name__ == '__main__':  # pragma: no cover
    unittest.main()
