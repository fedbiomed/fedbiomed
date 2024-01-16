import unittest
from unittest.mock import MagicMock, patch

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
#############################################################

import fedbiomed
from fedbiomed.researcher.federated_workflows import Experiment


class TestExperiment(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests")
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_experiment_01_initialization(self):
        # Experiment must be default-constructible
        exp = Experiment()
        self.assertIsInstance(exp.monitor(), fedbiomed.researcher.monitor.Monitor)  # set by default
        self.assertIsInstance(exp.aggregator(), fedbiomed.researcher.aggregators.FedAverage)  # set by default
        # aggregator
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = 'mock-aggregator'
        exp.set_aggregator(_aggregator)
        self.assertEqual(exp.aggregator(), _aggregator)
        _aggregator.set_fds.assert_called_once_with(exp.training_data())
        _aggregator_class = type('mock_aggregator_class',
                                 (fedbiomed.researcher.aggregators.Aggregator,),
                                 {'aggregator_name': 'mock-aggregator-2'})
        _aggregator_class_instance = MagicMock(spec=_aggregator_class)
        _aggregator_class.__call__ = lambda: _aggregator_class_instance
        exp.set_aggregator(_aggregator_class)
        self.assertEqual(exp.aggregator(), _aggregator_class_instance)
        _aggregator_class_instance.set_fds.assert_called_once_with(exp.training_data())

        # check that setting training data resets the aggregator's fds
        _aggregator.reset_mock()
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        exp.set_training_data(_training_data)
        _aggregator.set_fds.assert_called_once_with(exp.training_data())






if __name__ == '__main__':  # pragma: no cover
    unittest.main()
