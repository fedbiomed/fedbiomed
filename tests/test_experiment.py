import unittest
from itertools import product
from unittest.mock import create_autospec, MagicMock, patch

from declearn.model.api import Vector

from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedExperimentError
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.metrics import MetricTypes

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet

from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.node_state_agent import NodeStateAgent
from testsupport.base_mocks import MockRequestModule
from testsupport.fake_researcher_secagg import FakeSecAgg
from testsupport.fake_training_plan import (
    FakeTorchTrainingPlan,
    FakeSKLearnTrainingPlan,
)


from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet

from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.node_state_agent import NodeStateAgent
from declearn.optimizer.modules import AuxVar

import fedbiomed
from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedExperimentError
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers import AuxVar
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.node_state_agent import NodeStateAgent
from fedbiomed.researcher.secagg import SecureAggregation
from fedbiomed.researcher.secagg._secure_aggregation import _SecureAggregation
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


class TestExperiment(unittest.TestCase, MockRequestModule):
    def setUp(self):
        MockRequestModule.setUp(
            self,
            module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests",
        )
        super().setUp()

        self.patch_import_class_object = patch(
            "fedbiomed.researcher.federated_workflows._training_plan_workflow.import_class_object_from_file"
        )
        self.mock_import_class_object = self.patch_import_class_object.start()

        self.mock_tp = MagicMock()
        self.mock_import_class_object.return_value = None, self.mock_tp

        self.patch_job = patch(
            "fedbiomed.researcher.federated_workflows._experiment.TrainingJob"
        )
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
        self.assertIsInstance(
            exp.monitor(), fedbiomed.researcher.monitor.Monitor
        )  # set by default
        self.assertIsInstance(
            exp.aggregator(), fedbiomed.researcher.aggregators.FedAverage
        )  # set by default

        # Test all possible combinations of init arguments
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _secagg = MagicMock(spec=fedbiomed.researcher.secagg.SecureAggregation)
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = "mock aggregator"
        _strategy = MagicMock(spec=fedbiomed.researcher.strategies.Strategy)
        parameters_and_possible_values = {
            "training_data": (_training_data,),
            "training_args": (TrainingArgs({"epochs": 42}),),
            "secagg": (_secagg,),
            "save_breakpoints": (True, False),
            "training_plan_class": (
                FakeTorchTrainingPlan,
                FakeSKLearnTrainingPlan,
                None,
            ),
            "model_args": ({"model": "args"}, None),
            "aggregator": (_aggregator, None),
            "node_selection_strategy": (_strategy, None),
            "round_limit": (42, None),
            "tensorboard": (True, False),
            "retain_full_history": (True, False),
        }  # some of the parameters which have already been tested in FederatedWorkflow have been simplified here
        # Compute cartesian product of parameter values to obtain all possible combinations
        keys, values = zip(*parameters_and_possible_values.items())
        all_parameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
        for params in all_parameter_combinations:
            try:
                exp = Experiment(**params)
            except SystemExit as e:
                print(
                    f"Could not instantiate Experiment: exception {e} raised with the following constructor"
                    f"arguments:\n {params}"
                )
                raise e

    def test_experiment_02_set_aggregator(self):
        """Tests setting the Experiment's aggregator and related side effects"""
        exp = Experiment()
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        exp.set_training_data(_training_data)

        # test default case
        exp.set_aggregator()
        self.assertIsInstance(
            exp.aggregator(), fedbiomed.researcher.aggregators.FedAverage
        )
        self.assertEqual(exp.aggregator()._fds, _training_data)

        # setting through an object instance
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = "mock-aggregator"
        exp.set_aggregator(_aggregator)
        self.assertEqual(exp.aggregator(), _aggregator)
        _aggregator.set_fds.assert_called_once_with(
            exp.training_data()
        )  # side effect: aggregator's fds must be set to
        # be compatible with experiment fds

        # setting through a class
        _aggregator.reset_mock()

        class FakeAggregator(Aggregator):
            aggregator_name = "aggregator"
            is_set_fds_called = False

            def set_fds(self, fds: FederatedDataSet) -> FederatedDataSet:
                if not self.is_set_fds_called:
                    self.is_set_fds_called = True
                return super().set_fds(fds)

        _aggregator_class = FakeAggregator

        with self.assertRaises(SystemExit):
            exp.set_aggregator(_aggregator_class)

        exp.set_aggregator(FakeAggregator())
        self.assertIsInstance(exp.aggregator(), FakeAggregator)
        self.assertTrue(exp.aggregator().is_set_fds_called)
        self.assertEqual(exp.training_data().data(), exp.aggregator()._fds.data())

        # check that setting training data resets the aggregator's fds
        _aggregator.reset_mock()
        exp.set_aggregator(_aggregator)
        exp.set_training_data(_training_data)
        _aggregator.set_fds.assert_called_with(exp.training_data())
        self.assertEqual(_aggregator.set_fds.call_count, 2)

    def test_experiment_03_set_strategy(self):
        """Tests setting the Experiment's node selection strategy and related side effects"""
        exp = Experiment()

        # test default case
        exp.set_strategy()
        self.assertIsInstance(
            exp.strategy(),
            fedbiomed.researcher.strategies.default_strategy.DefaultStrategy,
        )

        # setting through an object instance
        _strategy = MagicMock(
            spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy
        )
        exp.set_strategy(_strategy)
        self.assertEqual(exp.strategy(), _strategy)

        # setting through a class
        _strategy.reset_mock()

        class FakeStrategy(DefaultStrategy):
            has_been_called = False

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.has_been_called = True

        _strategy_class = MagicMock()
        _strategy_class.return_value = _strategy

        with self.assertRaises(SystemExit):
            exp.set_strategy(FakeStrategy)

        exp.set_strategy(FakeStrategy())
        self.assertTrue(exp.strategy().has_been_called)

    #    @patch('fedbiomed.researcher.federated_workflows._experiment.TrainingJob')
    def test_experiment_04_run_once_base_case(self):
        """Tests run once method of experiment"""
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = "mock-aggregator"
        _strategy = MagicMock(
            spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy
        )
        _strategy.sample_nodes.return_value = ["node-1", "node-2"]
        _strategy.refine.return_value = (1, 2, 3, 4)

        exp = Experiment(
            training_data=_training_data,
            aggregator=_aggregator,
            round_limit=1,
            training_plan_class=FakeTorchTrainingPlan,
            node_selection_strategy=_strategy,
        )

        # Test error case -------------------------------
        with self.assertRaises(SystemExit):
            exp.run_once(increase="invalid-type")
        # ------------------------------------------------

        # Test if no training nodes are returned from strategy.sample_nodes ---
        _strategy.sample_nodes.return_value = []
        with self.assertRaises(SystemExit):
            exp.run_once()
        _strategy.sample_nodes.return_value = ["node-1", "node-2"]
        # ---------------------------------------------------------------------

        # Go back to normal
        exp._round_limit = 1
        exp._current_round = 0

        # Check if it raises if there is missing object block --------
        exp._fds = None
        with self.assertRaises(SystemExit):
            exp.run_once()
        exp._fds = _training_data
        # -------------------------------------------------------------

        # Run experiment ----------------------------------------------
        _strategy.sample_nodes.reset_mock()
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

        # Test run_once with wrong round limit -----------------------------
        self.mock_job.return_value.reset_mock()
        exp.set_round_limit(1)
        x = exp.run_once(increase=False)
        self.assertEqual(x, 0)
        self.assertFalse(self.mock_job.return_value.execute.called)
        # ------------------------------------------------------------------

        # Check auto round limit increase -----------------------------
        exp._round_current = 2
        exp.run_once(increase=True)
        self.assertEqual(exp._round_limit, 3)
        # ------------------------------------------------------------

        # Run once with secure aggregation ----------------------------------------
        secagg = MagicMock(spec=_SecureAggregation, instance=True)
        type(secagg).active = True
        # type(secagg).return_value = MagicMock(spec=_SecureAggregation)
        exp.set_round_limit(6)
        exp._secagg = secagg
        exp.run_once()
        exp.set_secagg(False)
        # -------------------------------------------------------------------------

        # Run once with whereas breakpoint is active
        with patch(
            "fedbiomed.researcher.federated_workflows.Experiment.breakpoint"
        ) as m_breakpoint:
            exp.set_save_breakpoints(True)
            exp.run_once()
            m_breakpoint.assert_called_once()
            exp.set_save_breakpoints(False)
        # -------------------------------------------------------------------------

        # Test run_once with test_after -------------------------------------------
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
        # 4 replies also from previous executions
        self.assertEqual(
            len(exp.training_replies()), 5
        )  # validation replies are not saved
        _strategy.refine.assert_called_once()
        _aggregator.aggregate.assert_called_once()
        exp.training_plan().set_model_params.assert_called_once()
        self.assertEqual(exp.round_current(), 6)
        # ------------------------------------------------------------------------

    def test_experiment_05_run(self):
        """Tests run method of the experiment"""
        exp = Experiment(round_limit=2)

        # check that round limit is reached when run is called without arguments
        with patch.object(exp, "run_once", return_value=1) as mock_run_once:
            exp.run()
            self.assertEqual(mock_run_once.call_count, 2)
            exp._set_round_current(2)  # manually set because of patched run_once

        # check that we can dynamically increase the round limit
        exp.set_round_limit(exp.round_current() + 5)
        self.assertEqual(exp.round_limit(), 2 + 5)
        with patch.object(exp, "run_once", return_value=1) as mock_run_once:
            exp.run()
            self.assertEqual(mock_run_once.call_count, 5)
            exp._set_round_current(2 + 5)  # manually set because of patched run_once

        # check that increase=True auto-increases the round limit
        with patch.object(exp, "run_once", return_value=1) as mock_run_once:
            exp.run(rounds=1, increase=True)
            self.assertEqual(mock_run_once.call_count, 1)
            self.assertEqual(exp.round_limit(), 2 + 5 + 1)
            exp._set_round_current(
                2 + 5 + 1
            )  # manually set because of patched run_once

        # check that increase=False takes precedence over user-specified rounds
        exp.set_round_limit(exp.round_current() + 2)
        with patch.object(exp, "run_once", return_value=1) as mock_run_once:
            exp.run(
                rounds=100, increase=False
            )  # should only run for two rounds because of round_limit
            self.assertEqual(mock_run_once.call_count, 2)
            self.assertEqual(exp.round_limit(), 2 + 5 + 1 + 2)
            exp._set_round_current(
                2 + 5 + 1 + 2
            )  # manually set because of patched run_once

        # wrong argument types
        with patch.object(exp, "run_once", return_value=1) as mock_run_once:
            with self.assertRaises(SystemExit):
                exp.run(rounds=-1)
                self.assertFalse(mock_run_once.called)
            with self.assertRaises(SystemExit):
                exp.run(rounds="one")
                self.assertFalse(mock_run_once.called)
            with self.assertRaises(SystemExit):
                exp.run(increase="True")
                self.assertFalse(mock_run_once.called)

        # inconsistent arguments
        with patch.object(exp, "run_once", return_value=1) as mock_run_once:
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
            exp.set_training_args({"test_on_global_updates": True})
            exp.set_round_limit(exp.round_current() + 1)
            x = exp.run()
            self.assertEqual(x, 1)
            mock_run_once.assert_called_once_with(increase=False, test_after=True)

        # Tests if run once return 0
        with patch.object(exp, "run_once", return_value=0) as mock_run_once:
            with self.assertRaises(SystemExit):
                exp.set_round_limit(exp.round_current() + 1)
                x = exp.run()

    def test_experiment_06_run_once_special_cases(self):
        """Tests running experiment special casses"""

        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = "mock-aggregator"
        _aggregator.create_aggregator_args.return_value = ({}, {})
        _strategy = MagicMock(
            spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy
        )
        _strategy.sample_nodes.return_value = ["node-1", "node-2"]
        _strategy.refine.return_value = (1, 2, 3, 4)

        # Test using aggregator-level optimizer
        _agg_optim = MagicMock(spec=fedbiomed.common.optimizers.Optimizer)
        exp = Experiment(
            training_data=_training_data,
            aggregator=_aggregator,
            round_limit=1,
            training_plan_class=FakeTorchTrainingPlan,
            node_selection_strategy=_strategy,
            agg_optimizer=_agg_optim,
        )
        with patch.object(Vector, "build", new=create_autospec(Vector)):
            exp.run_once()
        self.assertListEqual(
            [name for name, *_ in _agg_optim.method_calls],
            ["get_aux", "send_to_device", "init_round", "step"],
            "Aggregator optimizer did not receive expected ordered calls",
        )

        # Test that receiving auxiliary variables without an aggregator-level optimizer fails
        self.mock_job.reset_mock()
        mock_aux_var = create_autospec(AuxVar, instance=True)
        self.mock_job.return_value.execute.return_value = (
            MagicMock(),
            {"node_id": {"module": mock_aux_var}},
        )

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
            error_msg.startswith("Received auxiliary variables from 1+ node Optimizer"),
            "Receiving un-processable auxiliary variables did not raise "
            "the excepted exception.",
        )

    @patch("builtins.eval")
    @patch("builtins.print")
    def test_experiment_07_info(self, mock_print, mock_eval):
        exp = Experiment()
        _ = exp.info()

    def test_experiment_08_save_training_replies(self):
        """Test info method of the experiment"""

        exp = Experiment()
        exp._training_replies = {
            0: {"node1": {"params": "params", "other": "metadata"}},
            1: {"node1": {"only": "metadata"}},
        }

        replies_to_save = exp.save_training_replies()
        self.assertDictEqual(
            replies_to_save,
            {0: {"node1": {"other": "metadata"}}, 1: {"node1": {"only": "metadata"}}},
        )

    @patch(
        "fedbiomed.researcher.federated_workflows._experiment.TrainingPlanWorkflow.breakpoint"
    )
    @patch(
        "fedbiomed.researcher.federated_workflows._experiment.uuid.uuid4",
        return_value="UUID",
    )
    @patch(
        "fedbiomed.researcher.federated_workflows._experiment.choose_bkpt_file",
        return_value=("/bkpt-path", "bkpt-folder"),
    )
    @patch("fedbiomed.researcher.federated_workflows._experiment.Serializer")
    def test_experiment_09_breakpoint(
        self,
        mock_serializer,
        mock_choose_bkpt_file,
        mock_uuid,
        mock_super_breakpoint,
    ):
        # define attributes that will be saved in breakpoint
        _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
        _aggregator.aggregator_name = "mock-aggregator"
        agg_bkpt = {"agg": "bkpt"}
        _aggregator.save_state_breakpoint.return_value = agg_bkpt

        _agg_optim = MagicMock(spec=fedbiomed.common.optimizers.Optimizer)

        _strategy = MagicMock(
            spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy
        )
        strat_bkpt = {"strategy": "bkpt"}
        _strategy.save_state_breakpoint.return_value = strat_bkpt

        exp = Experiment(
            round_limit=5,
            aggregator=_aggregator,
            agg_optimizer=_agg_optim,
            node_selection_strategy=_strategy,
        )

        # Test if current round is less than 1 meaning that there is no round ran
        with self.assertRaises(SystemExit):
            exp._set_round_current(0)
            exp.breakpoint()
        # ----------------------------------------------------------------------

        exp._set_round_current(2)
        with (
            patch.object(
                exp, "save_aggregated_params", return_value={"agg_params": "bkpt"}
            ) as mock_agg_param_save,
            patch.object(
                exp, "save_training_replies", return_value={"replies": "bkpt"}
            ) as mock_save_replies,
            patch.object(exp, "training_plan") as mock_tp,
        ):
            exp.breakpoint()

        # This also validates the breakpoint scheme: if this fails, please consider updating the breakpoints version
        mock_super_breakpoint.assert_called_once_with(
            {
                "round_current": 2,
                "round_limit": 5,
                "aggregator": agg_bkpt,
                "agg_optimizer": "/bkpt-path/optimizer_UUID.mpk",
                "node_selection_strategy": strat_bkpt,
                "aggregated_params": {"agg_params": "bkpt"},
                "training_replies": {"replies": "bkpt"},
            },
            2,
        )

    @patch(
        "fedbiomed.researcher.federated_workflows._experiment.TrainingPlanWorkflow.load_breakpoint"
    )
    @patch("fedbiomed.researcher.federated_workflows._experiment.Serializer")
    @patch.object(
        fedbiomed.researcher.federated_workflows._experiment.Optimizer,
        "load_state",
        return_value=MagicMock(spec=fedbiomed.common.optimizers.Optimizer),
    )
    def test_experiment_10_load_breakpoint(
        self,
        mock_optimizer,
        mock_serializer,
        mock_super_load,
    ):
        mock_super_load.return_value = (
            Experiment(),
            {
                "round_current": 2,
                "round_limit": 5,
                "aggregator": {"agg": "bkpt"},
                "agg_optimizer": "/bkpt-path/optimizer_UUID.mpk",
                "node_selection_strategy": {"strategy": "bkpt"},
                "aggregated_params": {0: {"params_path": "bkpt"}},
                "training_replies": {
                    0: {"node1": {"params_path": "bkpt", "other": "reply-data"}}
                },
            },
        )

        def _gen():
            _strategy = MagicMock(
                spec=fedbiomed.researcher.strategies.default_strategy.DefaultStrategy
            )
            yield _strategy
            _aggregator = MagicMock(spec=fedbiomed.researcher.aggregators.Aggregator)
            _aggregator.aggregator_name = "mock-aggregator"
            yield _aggregator

        _g = _gen()

        def create_strategy_then_aggregator(*args, **kwargs):
            for x in _g:
                return x

        with patch.object(
            Experiment, "_create_object", new=create_strategy_then_aggregator
        ) as mock_creat:
            exp = Experiment.load_breakpoint()

    def test_experiment_11_testing_args(self):
        """Tests training arguments setter and getter"""

        exp = Experiment()
        exp.set_training_args(
            {
                "test_ratio": 0.2,
                "test_on_local_updates": True,
                "test_on_global_updates": True,
                "test_metric": MetricTypes.ACCURACY,
            }
        )

        self.assertEqual(exp.test_ratio(), (0.2, False))
        self.assertEqual(exp.test_metric(), MetricTypes.ACCURACY)
        self.assertDictEqual(exp.test_metric_args(), {})
        self.assertEqual(exp.test_on_local_updates(), True)
        self.assertEqual(exp.test_on_global_updates(), True)

        exp.set_test_ratio(0.2)
        self.assertEqual(exp.test_ratio(), (0.2, False))

        _, _ = exp.set_test_metric(MetricTypes.ACCURACY, test=1)
        self.assertDictEqual(exp.test_metric_args(), {"test": 1})
        self.assertEqual(exp.test_metric(), MetricTypes.ACCURACY)

        exp.set_test_on_local_updates(True)
        self.assertTrue(exp.test_on_local_updates())

        exp.set_test_on_global_updates(False)
        self.assertFalse(exp.test_on_global_updates())

    def test_experiment_12_set_agg_optimizer(self):
        """Tests setting aggregator optimizer"""

        exp = Experiment()
        _agg_optim = MagicMock(spec=fedbiomed.common.optimizers.Optimizer)

        # Set optimizer
        exp.set_agg_optimizer(_agg_optim)
        self.assertEqual(exp._agg_optimizer, _agg_optim)

        # Set None
        exp.set_agg_optimizer(None)
        self.assertIsNone(exp._agg_optimizer)

        invalid_type_optimizer = MagicMock()
        with self.assertRaises(SystemExit):
            exp.set_agg_optimizer(invalid_type_optimizer)

        # Check getter
        exp._agg_optimizer = _agg_optim
        self.assertEqual(exp.agg_optimizer(), _agg_optim)

    def test_experiment_13_set_round_limit(self):
        """Tests setting round limit"""

        # Normal good case
        exp = Experiment()
        limit = exp.set_round_limit(2)
        self.assertEqual(exp._round_limit, 2)

        # Test if round limit less than current round number
        exp._round_current = 2
        with self.assertRaises(SystemExit):
            limit = exp.set_round_limit(1)

        # Test setting it to None
        exp.set_round_limit(None)
        self.assertIsNone(exp._round_limit)

    def test_experiment_14_set_round_current(self):
        """Tests setting current round during the experiment"""

        # Normal case
        exp = Experiment()
        exp._set_round_current(2)
        self.assertEqual(exp.round_current(), 2)

        # Exception if round limit is less than round current
        exp._round_limit = 4
        with self.assertRaises(SystemExit):
            exp._set_round_current(5)

        # Check monitor called correctly
        exp._monitor = MagicMock(spec=Monitor)
        exp._set_round_current(2)
        exp._monitor.set_round.assert_called_once_with(2 + 1)

    def test_experiment_15_set_tensorboard(self):
        """Tests setting tensorboard"""

        exp = Experiment()
        exp._monitor = MagicMock(spec=Monitor)
        exp.set_tensorboard(True)
        exp._monitor.set_tensorboard.assert_called_once_with(True)

        # Invalid type
        with self.assertRaises(SystemExit):
            exp.set_tensorboard("opps")

    def test_experiment_16_set_retain_full_history(self):
        """Tests setting retain full history"""

        exp = Experiment()
        exp.set_retain_full_history(True)
        self.assertTrue(exp.retain_full_history())

        # Tests setting invalid type
        with self.assertRaises(SystemExit):
            exp.set_retain_full_history("invalid_type")

    def test_experiment_17_commit_experiment_history(self):
        """Tests commit experiment history"""

        exp = Experiment()
        exp.commit_experiment_history({"reply": 1}, {"param": 1})

        self.assertDictEqual(exp._training_replies[0], {"reply": 1})
        self.assertDictEqual(exp._aggregated_params[0], {"params": {"param": 1}})

        exp._retain_full_history = False
        exp._round_current = 5
        exp.commit_experiment_history({"reply": 1}, {"param": 1})

        self.assertDictEqual(exp.training_replies(), {5: {"reply": 1}})
        self.assertDictEqual(exp.aggregated_params(), {5: {"params": {"param": 1}}})

    @patch("fedbiomed.researcher.federated_workflows._experiment.Serializer")
    def test_experiment_18_load_training_replies(self, serializer):
        """Tests loading training replies"""

        serializer.load.return_value = {"p": 1}
        exp = Experiment()

        # Test empty training reply
        exp.load_training_replies({})

        # Test normal case ---------------------------------------------------------
        reply = {
            0: {"node-1": {"params_path": "x"}, "node-2": {"params_path": "x"}},
            1: {"node-1": {"params_path": "x"}, "node-2": {"params_path": "x"}},
        }

        exp.load_training_replies(reply)
        self.assertDictEqual(
            exp._training_replies,
            {
                0: {
                    "node-1": {"params_path": "x", "params": {"p": 1}},
                    "node-2": {"params_path": "x", "params": {"p": 1}},
                },
                1: {
                    "node-1": {"params_path": "x", "params": {"p": 1}},
                    "node-2": {"params_path": "x", "params": {"p": 1}},
                },
            },
        )

    def test_experiment_19_save_aggregated_params(self):
        """Tests saving aggregated params"""

        # Prepares arguments
        path = "path/params"
        params = {0: {"params": 1}, 1: {"params": 1}}
        exp = Experiment()

        # Error case invalid aggregated params
        with self.assertRaises(SystemExit):
            exp.save_aggregated_params("boom", path)

        # Error case invalid path
        with self.assertRaises(SystemExit):
            exp.save_aggregated_params(params, True)

        # Error case invalid params
        with self.assertRaises(SystemExit):
            exp.save_aggregated_params({"x": 1}, path)

        with patch(
            "fedbiomed.researcher.federated_workflows._experiment.Serializer"
        ) as ser:
            aggregated_params = exp.save_aggregated_params(params, path)

        self.assertTrue(0 in aggregated_params)
        self.assertTrue(1 in aggregated_params)
        self.assertIsInstance(aggregated_params[0]["params_path"], str)
        self.assertIsInstance(aggregated_params[1]["params_path"], str)

    def test_experiment_20_load_aggregated_params(self):
        """Tests loading aggregated paramaters"""

        exp = Experiment()

        # Invalid type of aggregated params
        with self.assertRaises(SystemExit):
            exp._load_aggregated_params(None)

        with patch(
            "fedbiomed.researcher.federated_workflows._experiment.Serializer"
        ) as ser:
            ser.load.return_value = 0
            agg_params = {"0": {"params_path": 1}, "1": {"params_path": 1}}
            agg_params_final = exp._load_aggregated_params(agg_params)

        self.assertTrue(agg_params_final[0]["params"] == 0)
        self.assertTrue(agg_params_final[1]["params"] == 0)

    def test_experiment_21_save_optimizer(self):
        """Tests saving optimizer"""

        exp = Experiment()
        # Check if agg optimzer is None
        result = exp.save_optimizer("my-path")
        self.assertIsNone(result)

        # Normal case
        with patch(
            "fedbiomed.researcher.federated_workflows._experiment.Serializer"
        ) as ser:
            ser.dump.return_value = "hello"
            exp._agg_optimizer = MagicMock(spec=fedbiomed.common.optimizers.Optimizer)
            r = exp.save_optimizer("your-path")
            self.assertTrue("optimizer_" in r)

    def test_experiment_22_load_optimizer(self):
        """Tests loading experiment"""

        path = "sate-path"

        exp = Experiment()
        r = exp._load_optimizer(None)
        self.assertIsNone(r)

        with (
            patch(
                "fedbiomed.researcher.federated_workflows._experiment.Optimizer"
            ) as opt,
            patch(
                "fedbiomed.researcher.federated_workflows._experiment.Serializer"
            ) as ser,
        ):
            exp._load_optimizer("state-path")
            opt.load_state.assert_called_once()

    def test_experiment_23_update_nodes_states_agent(self):
        """Tests updating node state agent"""

        state_agent = MagicMock(spec=NodeStateAgent)
        exp = Experiment()
        exp._node_state_agent = state_agent

        with patch.object(exp, "all_federation_nodes") as afn:
            afn.return_value = ["node-1", "node-2"]
            exp._update_nodes_states_agent(before_training=True, training_replies=None)
            state_agent.update_node_states.assert_called_once()

        # Test if training replies None while before_training=False
        with self.assertRaises(FedbiomedValueError):
            exp._update_nodes_states_agent(False, None)

        # Test normal case
        state_agent.reset_mock()
        with patch.object(exp, "all_federation_nodes") as afn:
            afn.return_value = ["node-1", "node-2"]
            exp._update_nodes_states_agent(False, {"hello": "world"})
            state_agent.update_node_states.assert_called_once_with(
                ["node-1", "node-2"], {"hello": "world"}
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
