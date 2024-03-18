import unittest
from itertools import product
from unittest.mock import ANY, create_autospec, MagicMock, patch

from declearn.model.api import Vector

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
from testsupport.fake_training_plan import (
    FakeTorchTrainingPlan,
    FakeSKLearnTrainingPlan
)
#############################################################

import fedbiomed
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob


class TestExperiment(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests")
        super().setUp()

        self.patch_import_class_object = patch(
            'fedbiomed.researcher.federated_workflows._training_plan_workflow.import_class_object_from_file'
        )
        self.mock_import_class_object = self.patch_import_class_object.start()

        self.mock_tp = MagicMock()
        self.mock_import_class_object.return_value = None, self.mock_tp


        self.patch_job = patch('fedbiomed.researcher.federated_workflows._experiment.TrainingJob')
        self.mock_job = self.patch_job.start()
        self.mock_job.return_value = MagicMock(spec=TrainingJob)
        self.mock_job.return_value._training_replies = {}

        self.mock_job.return_value.execute.return_value = MagicMock(), {}

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
            'training_data': (_training_data,),
            'training_args': (TrainingArgs({'epochs': 42}),),
            'secagg': (_secagg,),
            'save_breakpoints': (True, False),
            'training_plan_class': (
                FakeTorchTrainingPlan,
                FakeSKLearnTrainingPlan,
                None
            ),
            'model_args': ({'model': 'args'}, None),
            'aggregator': (_aggregator, None),
            'node_selection_strategy': (_strategy, None),
            'round_limit': (42, None),
            'tensorboard': (True, False),
            'retain_full_history': (True, False)
        }  # some of the parameters which have already been tested in FederatedWorkflow have been simplified here
        # Compute cartesian product of parameter values to obtain all possible combinations
        keys, values = zip(*parameters_and_possible_values.items())
        all_parameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
        for params in all_parameter_combinations:
            try:
                exp = Experiment(**params)
            except SystemExit as e:
                print(f'Could not instantiate Experiment: exception {e} raised with the following constructor'
                      f'arguments:\n {params}')
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

#    @patch('fedbiomed.researcher.federated_workflows._experiment.TrainingJob')
    def test_experiment_04_run_once_base_case(self):

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
        # 4. call Job's execute()
        self.mock_job.return_value.execute.assert_called_once()
        # 5. populate training replies
        self.assertEqual(len(exp.training_replies()), 1)
        # 6. node strategy refine
        _strategy.refine.assert_called_once()
        # 7. aggregate
        _aggregator.aggregate.assert_called_once()
        # 8. update training plan with aggregated params
        exp.training_plan().set_model_params.assert_called_once()
        # 9. increase round counter
        self.assertEqual(exp.round_current(), 1)

        # Test run_once with wrong round limit
        self.mock_job.return_value.reset_mock()
        exp.set_round_limit(1)
        x = exp.run_once(increase=False)
        self.assertEqual(x, 0)
        self.assertFalse(self.mock_job.return_value.execute.called)

        # Test run_once with test_after
        self.mock_job.reset_mock()
        _strategy.reset_mock()
        _aggregator.reset_mock()
        exp.training_plan().reset_mock()
        exp.set_round_limit(exp.round_current() + 1)
        x = exp.run_once(increase=False, test_after=True)
        # check that everything ran as expected
        self.assertEqual(x, 1)
        _strategy.sample_nodes.assert_called_once()
        self.assertEqual(exp.training_plan().after_training_params.call_count, 2)
        self.assertEqual(_aggregator.create_aggregator_args.call_count, 2)
        self.assertEqual(self.mock_job.return_value.execute.call_count, 2)
        self.assertEqual(len(exp.training_replies()), 2)  # validation replies are not saved
        _strategy.refine.assert_called_once()
        _aggregator.aggregate.assert_called_once()
        exp.training_plan().set_model_params.assert_called_once()
        self.assertEqual(exp.round_current(), 2)

    def test_experiment_05_run(self):
        exp = Experiment(round_limit=2)

        # check that round limit is reached when run is called without arguments
        with patch.object(exp, 'run_once', return_value=1) as mock_run_once:
            exp.run()
            self.assertEqual(mock_run_once.call_count, 2)
            exp._set_round_current(2)  # manually set because of patched run_once

        # check that we can dynamically increase the round limit
        exp.set_round_limit(exp.round_current() + 5)
        self.assertEqual(exp.round_limit(), 2+5)
        with patch.object(exp, 'run_once', return_value=1) as mock_run_once:
            exp.run()
            self.assertEqual(mock_run_once.call_count, 5)
            exp._set_round_current(2+5)  # manually set because of patched run_once

        # check that increase=True auto-increases the round limit
        with patch.object(exp, 'run_once', return_value=1) as mock_run_once:
            exp.run(rounds=1, increase=True)
            self.assertEqual(mock_run_once.call_count, 1)
            self.assertEqual(exp.round_limit(), 2+5+1)
            exp._set_round_current(2+5+1)  # manually set because of patched run_once

        # check that increase=False takes precedence over user-specified rounds
        exp.set_round_limit(exp.round_current() + 2)
        with patch.object(exp, 'run_once', return_value=1) as mock_run_once:
            exp.run(rounds=100, increase=False)  # should only run for two rounds because of round_limit
            self.assertEqual(mock_run_once.call_count, 2)
            self.assertEqual(exp.round_limit(), 2+5+1+2)
            exp._set_round_current(2+5+1+2)  # manually set because of patched run_once

        # wrong argument types
        with patch.object(exp, 'run_once', return_value=1) as mock_run_once:
            with self.assertRaises(SystemExit):
                exp.run(rounds=0)
                self.assertFalse(mock_run_once.called)
            with self.assertRaises(SystemExit):
                exp.run(rounds='one')
                self.assertFalse(mock_run_once.called)
            with self.assertRaises(SystemExit):
                exp.run(increase='True')
                self.assertFalse(mock_run_once.called)

        # inconsistent arguments
        with patch.object(exp, 'run_once', return_value=1) as mock_run_once:
            for increase in (True, False):
                x = exp.run(increase=increase)
                self.assertFalse(mock_run_once.called)
                self.assertEqual(x, 0)

            exp.set_round_limit(None)
            for increase in (True, False):
                x = exp.run(increase=increase)
                self.assertFalse(mock_run_once.called)
                self.assertEqual(x, 0)

            # trying to run for more rounds than round_limit
            exp.set_round_limit(exp.round_current())
            x = exp.run(rounds=2, increase=False)
            self.assertFalse(mock_run_once.called)
            self.assertEqual(x, 0)

            # check that one last validation round is performed when test_on_global_updates is True
            exp.set_training_args({'test_on_global_updates': True})
            exp.set_round_limit(exp.round_current() + 1)
            x = exp.run()
            self.assertEqual(x, 1)
            mock_run_once.assert_called_once_with(increase=False, test_after=True)

    def test_experiment_06_run_once_special_cases(self):

        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = 'mock-aggregator'
        _aggregator.create_aggregator_args.return_value = ({}, {})
        _strategy = MagicMock(spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy)
        _strategy.refine.return_value = (1, 2, 3, 4)

        # Test using aggregator-level optimizer
        _agg_optim = MagicMock(spec=fedbiomed.common.optimizers.Optimizer)
        exp = Experiment(
            training_data=_training_data,
            aggregator=_aggregator,
            round_limit=1,
            training_plan_class=FakeTorchTrainingPlan,
            node_selection_strategy=_strategy,
            agg_optimizer=_agg_optim
        )
        with patch.object(Vector, "build", new=create_autospec(Vector)):
            exp.run_once()
        self.assertListEqual(
            [name for name, *_ in _agg_optim.method_calls],
            ["get_aux", "set_aux", "init_round", "step"],
            "Aggregator optimizer did not receive expected ordered calls"
        )

        # Test that receiving auxiliary variables without an aggregator-level optimizer fails
        self.mock_job.reset_mock()
        self.mock_job.return_value.execute.return_value = MagicMock(), {"module": {"node_id": {"key": "val"}}}  # mock aux-var dict

        exp = Experiment(
            training_data=_training_data,
            aggregator=_aggregator,
            round_limit=1,
            training_plan_class=FakeTorchTrainingPlan,
            node_selection_strategy=_strategy,
        )
        with patch.object(FedbiomedExperimentError, "__init__") as patch_exc:
            patch_exc.return_value = None  # __init__ must return None
            self.assertRaises(SystemExit, exp.run_once)
        patch_exc.assert_called_once()
        error_msg = patch_exc.call_args[0][0]
        self.assertTrue(
            error_msg.startswith(
                "Received auxiliary variables from 1+ node Optimizer"
            ),
            "Receiving un-processable auxiliary variables did not raise "
            "the excepted exception."
        )

    @patch('builtins.eval')
    @patch('builtins.print')
    def test_experiment_07_info(self,
                                mock_print,
                                mock_eval):
        exp = Experiment()
        _ = exp.info()

    def test_experiment_08_save_training_replies(self):
        exp = Experiment()
        exp._training_replies = {
            0: {
                'node1': {
                    'params': 'params',
                    'other': 'metadata'
                }
            },
            1 : {
                'node1': {
                    'only': 'metadata'
                }
            }
        }

        replies_to_save = exp.save_training_replies()
        self.assertDictEqual(
            replies_to_save,
            {
                0: {
                    'node1': {
                        'other': 'metadata'
                    }
                },
                1 : {
                    'node1': {
                        'only': 'metadata'
                    }
                }
            }
        )

    @patch('fedbiomed.researcher.federated_workflows._experiment.TrainingPlanWorkflow.breakpoint')
    @patch('fedbiomed.researcher.federated_workflows._experiment.uuid.uuid4', return_value='UUID')
    @patch('fedbiomed.researcher.federated_workflows._experiment.choose_bkpt_file',
           return_value=('/bkpt-path', 'bkpt-folder'))
    @patch('fedbiomed.researcher.federated_workflows._experiment.Serializer')
    def test_experiment_09_breakpoint(self,
                                      mock_serializer,
                                      mock_choose_bkpt_file,
                                      mock_uuid,
                                      mock_super_breakpoint,
                                      ):
        # define attributes that will be saved in breakpoint
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = 'mock-aggregator'
        agg_bkpt = {"agg": 'bkpt'}
        _aggregator.save_state_breakpoint.return_value = agg_bkpt

        _agg_optim = MagicMock(spec=fedbiomed.common.optimizers.Optimizer)

        _strategy = MagicMock(spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy)
        strat_bkpt = {'strategy': 'bkpt'}
        _strategy.save_state_breakpoint.return_value = strat_bkpt

        exp = Experiment(
            round_limit=5,
            aggregator=_aggregator,
            agg_optimizer=_agg_optim,
            node_selection_strategy=_strategy
        )
        exp._set_round_current(2)

        with patch.object(exp, 'save_aggregated_params', return_value={'agg_params': 'bkpt'}) as mock_agg_param_save,\
                patch.object(exp, 'save_training_replies', return_value={'replies': 'bkpt'}) as mock_save_replies, \
                patch.object(exp, 'training_plan') as mock_tp:
            exp.breakpoint()

        # This also validates the breakpoint scheme: if this fails, please consider updating the breakpoints version
        mock_super_breakpoint.assert_called_once_with(
            {
                'round_current': 2,
                'round_limit': 5,
                'aggregator': agg_bkpt,
                'agg_optimizer': '/bkpt-path/optimizer_UUID.mpk',
                'node_selection_strategy': strat_bkpt,
                'aggregated_params': {'agg_params': 'bkpt'},
                'training_replies': {'replies': 'bkpt'},
            },
            2
        )

    @patch('fedbiomed.researcher.federated_workflows._experiment.TrainingPlanWorkflow.load_breakpoint')
    @patch('fedbiomed.researcher.federated_workflows._experiment.Serializer')
    @patch.object(fedbiomed.researcher.federated_workflows._experiment.Optimizer, 'load_state',
                  return_value=MagicMock(spec=fedbiomed.common.optimizers.Optimizer))
    def test_federated_workflow_06_load_breakpoint(self,
                                                   mock_optimizer,
                                                   mock_serializer,
                                                   mock_super_load,
                                                   ):
        mock_super_load.return_value = (
            Experiment(),
            {
                'round_current': 2,
                'round_limit': 5,
                'aggregator': {"agg": 'bkpt'},
                'agg_optimizer': '/bkpt-path/optimizer_UUID.mpk',
                'node_selection_strategy': {'strategy': 'bkpt'},
                'aggregated_params': {0: {'params_path': 'bkpt'}},
                'training_replies': {0: {'node1': {'params_path': 'bkpt', 'other': 'reply-data'}}},
            }
        )

        def _gen():
            _strategy = MagicMock(spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy)
            yield _strategy
            _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
            _aggregator.aggregator_name = 'mock-aggregator'
            yield _aggregator
        _g = _gen()
        def create_strategy_then_aggregator(*args, **kwargs):
            for x in _g:
                return x

        with patch.object(Experiment, '_create_object', new=create_strategy_then_aggregator) as mock_creat:
            exp = Experiment.load_breakpoint()




if __name__ == '__main__':  # pragma: no cover
    unittest.main()
