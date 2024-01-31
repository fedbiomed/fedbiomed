import unittest
from itertools import product
from unittest.mock import MagicMock, patch

from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
from testsupport.fake_training_plan import FakeTorchTrainingPlan
#############################################################

import fedbiomed
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob


class TestExperiment(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests")
        super().setUp()
        self.patch_job = patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingJob')
        self.mock_job = self.patch_job.start()
        mock_training_job = MagicMock(spec=TrainingJob)
        self.mock_job.return_value = mock_training_job

    def tearDown(self):
        super().tearDown()
        self.patch_job.stop()

    def test_experiment_01_initialization(self):
        # Experiment must be default-constructible
        exp = Experiment()
        self.assertIsInstance(exp.monitor(), fedbiomed.researcher.monitor.Monitor)  # set by default
        self.assertIsInstance(exp.aggregator(), fedbiomed.researcher.aggregators.FedAverage)  # set by default

        # Test all possible combinations of init arguments
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _secagg = MagicMock(spec=fedbiomed.researcher.secagg.SecureAggregation)
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = 'mock aggregator'
        _strategy = MagicMock(spec=fedbiomed.researcher.strategies.Strategy)
        parameters_and_possible_values = {
            'tags': ('one-tag', ['one-tag', 'another-tag'], None),
            'nodes': (['one-node'], None),
            'training_data': (_training_data, {'one-node': {'tags': ['one-tag']}}, None),
            'training_args': (TrainingArgs({'epochs': 42}), {'num_updates': 1}, None),
            'experimentation_folder': ('folder_name', None),
            'secagg': (True, False, _secagg),
            'save_breakpoints': (True, False),
            'training_plan_class': (TorchTrainingPlan, SKLearnTrainingPlan, None),
            'model_args': ({'model': 'args'}, None),
            'aggregator': (_aggregator, None),
            'node_selection_strategy': (_strategy, None),
            'round_limit': (42, None),
            'tensorboard': (True, False)
        }
        # Compute cartesian product of parameter values to obtain all possible combinations
        keys, values = zip(*parameters_and_possible_values.items())
        all_parameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
        for params in all_parameter_combinations:
            try:
                exp = Experiment(**params)
            except SystemExit as e:
                print(f'Exception {e} raised with the following parameters {params}')
                raise e

    def test_experiment_02_set_aggregator(self):
        """Tests setting the Experiment's aggregator and related side effects"""
        exp = Experiment()
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        exp.set_training_data(_training_data)

        # test default case
        exp.set_aggregator()
        self.assertIsInstance(exp.aggregator(), fedbiomed.researcher.aggregators.FedAverage)
        self.assertEqual(exp.aggregator()._fds, _training_data)

        # setting through an object instance
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = 'mock-aggregator'
        exp.set_aggregator(_aggregator)
        self.assertEqual(exp.aggregator(), _aggregator)
        _aggregator.set_fds.assert_called_once_with(exp.training_data())  # side effect: aggregator's fds must be set to
                                                                          # be compatible with experiment fds

        # setting through a class
        _aggregator.reset_mock()
        _aggregator_class = MagicMock()
        _aggregator_class.return_value = _aggregator
        with patch('fedbiomed.researcher.federated_workflows._experiment.inspect.isclass', return_value=True), \
                patch('fedbiomed.researcher.federated_workflows._experiment.issubclass', return_value=True):
            exp.set_aggregator(_aggregator_class)
        self.assertEqual(exp.aggregator(), _aggregator)
        _aggregator.set_fds.assert_called_once_with(exp.training_data())  # same side effect

        # check that setting training data resets the aggregator's fds
        _aggregator.reset_mock()
        exp.set_training_data(_training_data)
        _aggregator.set_fds.assert_called_once_with(exp.training_data())

    def test_experiment_03_set_strategy(self):
        """Tests setting the Experiment's node selection strategy and related side effects"""
        exp = Experiment()

        # test default case
        exp.set_strategy()
        self.assertIsInstance(exp.strategy(), fedbiomed.researcher.strategies.default_strategy.DefaultStrategy)

        # setting through an object instance
        _strategy = MagicMock(spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy)
        exp.set_strategy(_strategy)
        self.assertEqual(exp.strategy(), _strategy)

        # setting through a class
        _strategy.reset_mock()
        _strategy_class = MagicMock()
        _strategy_class.return_value = _strategy
        with patch('fedbiomed.researcher.federated_workflows._experiment.inspect.isclass', return_value=True), \
                patch('fedbiomed.researcher.federated_workflows._experiment.issubclass', return_value=True):
            exp.set_strategy(_strategy_class)
        _strategy_class.assert_called_once_with()

    @patch('fedbiomed.researcher.federated_workflows._experiment.TrainingJob')
    def test_experiment_04_run_once_base_case(self,
                                              patch_train_job):
        patch_train_job.return_value.extract_received_optimizer_aux_var_from_round.return_value = {}
        self.mock_job.return_value.get_initialized_tp_instance.return_value = MagicMock(spec=FakeTorchTrainingPlan)

        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = 'mock-aggregator'
        _strategy = MagicMock(spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy)
        _strategy.refine.return_value = (1, 2, 3, 4)
        exp = Experiment(
            training_data=_training_data,
            aggregator=_aggregator,
            round_limit=1,
            training_plan_class=FakeTorchTrainingPlan,
            node_selection_strategy=_strategy
        )
        exp.run_once()
        # experiment run workflow:
        # 1. sample nodes
        _strategy.sample_nodes.assert_called_once()
        # 2. save model params before training
        exp.training_plan().after_training_params.assert_called_once()
        # 3. create aggregator arguments
        _aggregator.create_aggregator_args.assert_called_once()
        # 4. call Job's start_nodes_training_round
        patch_train_job.return_value.start_nodes_training_round.assert_called_once()
        # 5. populate training replies
        self.assertEqual(len(exp.training_replies()), 1)
        # 6. node strategy refine
        _strategy.refine.assert_called_once()
        # 7. aggregate
        _aggregator.aggregate.assert_called_once()
        # 8. update training plan with aggregated params
        exp.training_plan().set_model_params.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
