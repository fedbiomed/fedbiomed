import os
import unittest
import tempfile

from unittest.mock import MagicMock, patch


from testsupport.base_mocks import MockRequestModule

import fedbiomed
from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import (
    BaseTrainingPlan,
)
from fedbiomed.researcher.federated_workflows import TrainingPlanWorkflow
from fedbiomed.researcher.federated_workflows.jobs import (
    TrainingPlanApproveJob,
    TrainingPlanCheckJob,
)
from testsupport.fake_training_plan import (
    FakeTorchTrainingPlan,
    FakeSKLearnTrainingPlan,
)

from unittest.mock import ANY

from fedbiomed.researcher.config import config


class TestTrainingPlanWorkflow(unittest.TestCase, MockRequestModule):
    def setUp(self):
        MockRequestModule.setUp(
            self,
            module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests",
        )

        super().setUp()
        self.abstract_methods_patcher = patch.multiple(
            TrainingPlanWorkflow, __abstractmethods__=set()
        )
        self.abstract_methods_patcher.start()

        # Mock import class object from file for training plan class

        self.mock_tp = MagicMock(spec=FakeTorchTrainingPlan)
        self.patch_import_class_object = patch(
            "fedbiomed.researcher.federated_workflows._training_plan_workflow"
            ".import_class_object_from_file"
        )
        self.mock_import_class_object = self.patch_import_class_object.start()
        self.mock_import_class_object.return_value = None, self.mock_tp

        self.temp_dir = tempfile.TemporaryDirectory()
        config.load(root=self.temp_dir.name)

    def tearDown(self):
        super().tearDown()
        self.temp_dir.cleanup()
        self.abstract_methods_patcher.stop()
        self.patch_import_class_object.stop()

    def test_training_plan_workflow_01_initialization(self):
        """Test initialization of training plan workflow, only cases
        where correct parameters are provided"""

        def DummyMethod():
            pass

        with self.assertRaises(SystemExit):
            TrainingPlanWorkflow(training_plan_class=DummyMethod)

        class MyTrainingPlan(dict):
            pass

        with self.assertRaises(SystemExit):
            TrainingPlanWorkflow(training_plan_class=MyTrainingPlan)

        exp = TrainingPlanWorkflow()
        self.assertIsNone(exp.training_plan_class())
        self.assertIsNone(exp.model_args())
        self.assertIsNone(exp.training_plan())
        self.assertIsNotNone(exp.training_args())

        # Test all possible combinations of init arguments
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _secagg = MagicMock(spec=fedbiomed.researcher.secagg.SecureAggregation)
        parameters_and_possible_values = {
            "tags": (None, None, ["one-tag", "another-tag"]),
            "nodes": (["one-node"], None, None),
            "training_data": (
                _training_data,
                {"one-node": {"tags": ["one-tag"]}},
                None,
            ),
            "training_args": (TrainingArgs({"epochs": 42}), {"num_updates": 1}, None),
            "experimentation_folder": ("folder_name", None, None),
            "secagg": (True, False, _secagg),
            "save_breakpoints": (True, False, False),
            "training_plan_class": (
                FakeTorchTrainingPlan,
                FakeSKLearnTrainingPlan,
                None,
            ),
            "model_args": ({"model": "args"}, None, None),
        }
        # Compute cartesian product of parameter values to obtain all possible combinations
        combs = [
            {key: value[i] for key, value in parameters_and_possible_values.items()}
            for i in range(3)
        ]

        for params in combs:
            try:
                exp = TrainingPlanWorkflow(**params)
            except Exception as e:
                print(f"Exception {e} raised with the following parameters {params}")
                raise e

        # Special corner cases that deserve additional testing
        # TrainingPlanWorkflow can also be constructed by providing parameters to the constructor
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _training_data.node_ids.return_value = [
            "alice",
            "bob",
        ]  # make sure that nodes can be correctly inferred
        exp = TrainingPlanWorkflow(
            nodes=["alice", "bob"],
            training_data=_training_data,
            training_args={"num_updates": 1},
            secagg=True,
            save_breakpoints=True,
            training_plan_class=FakeTorchTrainingPlan,
            model_args={"model-args": "from-constructor"},
        )
        self.assertDictEqual(exp.model_args(), {"model-args": "from-constructor"})
        self.assertIsInstance(exp.training_plan(), FakeTorchTrainingPlan)
        # arguments only relevant to TrainingPlanWorkflow
        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan,
            model_args={"model-args": "from-constructor"},
            training_args={"num_updates": 1},
        )
        self.assertDictEqual(exp.model_args(), {"model-args": "from-constructor"})
        self.assertIsInstance(exp.training_plan(), FakeTorchTrainingPlan)

    def test_training_plan_workflow_02_set_training_plan_class(self):
        exp = TrainingPlanWorkflow()
        exp.set_training_plan_class(FakeTorchTrainingPlan)

        # check that weights are correctly preserved
        self.mock_tp.get_model_params.return_value = {"model": "params"}
        exp.set_training_plan_class(FakeTorchTrainingPlan)

        self.mock_tp.set_model_params.assert_called_once_with({"model": "params"})

        # check that weights are not preserved if we explicitly ask not to
        self.mock_tp.set_model_params.reset_mock()
        exp.set_training_plan_class(FakeTorchTrainingPlan, keep_weights=False)
        self.assertEqual(self.mock_tp.call_count, 0)

        # resetting training plan class to None
        exp.set_training_plan_class(None)
        self.assertIsNone(exp.training_plan_class())
        self.assertIsNone(exp.training_plan())

        # If training plan class is not subclass of training plan classes
        class DummyInvalidTPClass:
            pass

        with self.assertRaises(SystemExit):
            exp.set_training_plan_class(DummyInvalidTPClass)

        # Type invalid type of training plan class
        with self.assertRaises(SystemExit):
            exp.set_training_plan_class("Invalid")

    def test_training_plan_workflow_03_set_model_args(self):
        exp = TrainingPlanWorkflow()
        exp.set_training_plan_class(FakeTorchTrainingPlan)
        self.mock_tp.reset_mock()

        exp.set_model_args({"model": "args"}, keep_weights=False)
        self.assertDictEqual(exp.model_args(), {"model": "args"})

        self.mock_tp.post_init.assert_called_once_with(
            model_args={"model": "args"}, training_args=ANY, initialize_optimizer=False
        )

        # try to keep weights
        self.mock_tp.reset_mock()
        self.mock_tp.get_model_params.return_value = {"model": "new-params"}
        exp.set_model_args({"model": "other-args"})

        self.assertDictEqual(exp.model_args(), {"model": "other-args"})
        self.mock_tp.set_model_params.assert_called_once_with({"model": "new-params"})

        # Invalid model args type
        with self.assertRaises(SystemExit):
            exp.set_model_args("invalid-type")

    def test_training_plan_workflow_04_approval(self):
        """"""
        patch_job = patch(
            "fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingPlanApproveJob"
        )
        mock_job = patch_job.start()
        mock_approval_job = MagicMock(spec=TrainingPlanApproveJob)
        mock_job.return_value = mock_approval_job
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)

        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan, training_data=_training_data
        )

        response = exp.training_plan_approve(description="some description")
        mock_approval_job.execute.assert_called_once_with()

    def test_training_plan_workflow_05_status(self):
        """"""
        patch_job = patch(
            "fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingPlanCheckJob"
        )
        mock_job = patch_job.start()
        mock_approval_job = MagicMock(spec=TrainingPlanCheckJob)
        mock_job.return_value = mock_approval_job
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)

        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan, training_data=_training_data
        )

        status = exp.check_training_plan_status()
        mock_approval_job.execute.assert_called_once_with()

        # Error cases where training data is no
        exp._fds = None
        with self.assertRaises(SystemExit):
            exp.check_training_plan_status()

    @patch("fedbiomed.common.serializer.Serializer.dump")
    @patch(
        "fedbiomed.researcher.federated_workflows._training_plan_workflow.FederatedWorkflow.breakpoint"
    )
    @patch(
        "fedbiomed.researcher.federated_workflows._training_plan_workflow.uuid.uuid4",
        return_value="UUID",
    )
    def test_federated_workflow_06_breakpoint(
        self, mock_uuid, mock_super_breakpoint, mock_serializer_dump
    ):
        # define attributes that will be saved in breakpoint
        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan,
            training_args={"num_updates": 42},
            model_args={"breakpoint-model": "args"},
        )
        exp.breakpoint(state={}, bkpt_number=1)
        # Test if the Serializer.dump is called once with the good arguments
        params_path = os.path.join(
            exp.config.vars["EXPERIMENTS_DIR"],
            exp.experimentation_folder(),
            "breakpoint_0000",
            f"model_params_{mock_uuid.return_value}.mpk",
        )
        mock_serializer_dump.assert_called_once_with(
            self.mock_tp.get_model_wrapper_class.return_value.get_weights.return_value,
            params_path,
        )
        # This also validates the breakpoint scheme: if this fails, please consider updating the breakpoints version
        mock_super_breakpoint.assert_called_once_with(
            {
                "model_args": {"breakpoint-model": "args"},
                "training_plan_class_name": "FakeTorchTrainingPlan",
                "training_plan_path": os.path.join(
                    exp.config.vars["EXPERIMENTS_DIR"],
                    exp.experimentation_folder(),
                    "breakpoint_0000",
                    "model_0000.py",
                ),
                "model_weights_path": params_path,
                "training_args": TrainingArgs(
                    {"num_updates": 42}, only_required=False
                ).dict(),
            },
            1,
        )

    @patch(
        "fedbiomed.common.serializer.Serializer.load", return_value={"coefs": [1, 2, 3]}
    )
    @patch(
        "fedbiomed.researcher.federated_workflows."
        "_training_plan_workflow.import_class_from_file",
        return_value=(None, FakeTorchTrainingPlan),
    )
    @patch(
        "fedbiomed.researcher.federated_workflows."
        "_training_plan_workflow.FederatedWorkflow.load_breakpoint"
    )
    def test_federated_workflow_07_load_breakpoint(
        self, mock_super_load, mock_import_class, mock_serializer_load
    ):
        model_weights_path = "some-path-for-model-weights"
        mock_super_load.return_value = (
            TrainingPlanWorkflow(),
            {
                "model_args": {"breakpoint-model": "args"},
                "training_args": TrainingArgs(
                    {"num_updates": 42}, only_required=False
                ).dict(),
                "training_plan_class_name": "FakeTorchTrainingPlan",
                "training_plan_path": "some-path",
                "model_weights_path": model_weights_path,
            },
        )

        exp, saved_state = TrainingPlanWorkflow.load_breakpoint()
        # Test if Serializer.load is called once with the good arguments
        mock_serializer_load.assert_called_once_with(model_weights_path)
        self.assertEqual(exp.training_plan_class(), FakeTorchTrainingPlan)
        self.assertIsInstance(exp.training_plan(), FakeTorchTrainingPlan)
        self.assertDictEqual(exp.model_args(), {"breakpoint-model": "args"})
        self.assertDictEqual(
            exp.training_args(),
            TrainingArgs({"num_updates": 42}, only_required=False).dict(),
        )
        # Test if set_weights is called with the right arguments
        exp.training_plan().get_model_wrapper_class.return_value.set_weights.assert_called_once_with(
            mock_serializer_load.return_value
        )

    def test_training_plan_workflow_08_set_training_args(self):
        """Tests setting training arguments"""
        exp = TrainingPlanWorkflow()
        self.assertTrue(isinstance(exp.training_args(), dict))
        self.assertTrue(len(exp.training_args()) >= 1)
        exp.set_training_args({"num_updates": 42})
        self.assertTrue(exp.training_args()["num_updates"] == 42)
        exp.set_training_args(TrainingArgs({"epochs": 42}))
        self.assertTrue(exp.training_args()["epochs"] == 42)

        # Invalid type of training args argument

        with self.assertRaises(SystemExit):
            exp.set_training_args(True)

    @patch("builtins.open")
    def test_training_plan_workflow_09_training_plan_file(self, mock_open):
        """Test training plan file method"""

        exp = TrainingPlanWorkflow()
        exp._training_plan_file = "path/to/training-plan.py"

        file = exp.training_plan_file()
        self.assertEqual(file, "path/to/training-plan.py")

        with self.assertRaises(SystemExit):
            file = exp.training_plan_file(display="invalid-type")

        file = exp.training_plan_file(display=True)
        mock_open.assert_called()

        mock_open.side_effect = OSError
        with self.assertRaises(SystemExit):
            file = exp.training_plan_file(display=True)

    def test_training_plan_workflow_10_info(self):
        """Covers training plan info method"""
        exp = TrainingPlanWorkflow()
        result = exp.info()
        self.assertIsNotNone(result)

    def test_training_plan_workflow_11_check_missing_objects(self):
        """Test training plan workflow object checks"""

        tpw = TrainingPlanWorkflow()
        result = tpw._check_missing_objects()
        self.assertTrue("Training Plan" in result)

    def test_training_plan_workflow_12_check_round_value_consistency(self):
        """Tests checking round value consistency"""

        exp = TrainingPlanWorkflow()

        with self.assertRaises(FedbiomedExperimentError):
            exp._check_round_value_consistency(
                round_current="invalid", variable_name="round"
            )

        with self.assertRaises(FedbiomedExperimentError):
            exp._check_round_value_consistency(round_current=-1, variable_name="round")

        result = exp._check_round_value_consistency(
            round_current=12, variable_name="round"
        )
        self.assertTrue(result)

    def test_training_plan_workflow_13_keep_weights(self):
        """Tests training plan keep weights functionality"""

        exp = TrainingPlanWorkflow()
        training_plan = MagicMock(spec=BaseTrainingPlan)
        exp._training_plan = training_plan

        def dummy_yield():
            pass

        # Check correct case
        with exp._keep_weights(True):
            dummy_yield()
        training_plan.set_model_params.assert_called_once()

        # Check set_model_params raises an exception

        training_plan.set_model_params.side_effect = Exception
        with self.assertRaises(FedbiomedExperimentError):
            with exp._keep_weights(True):
                dummy_yield()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
