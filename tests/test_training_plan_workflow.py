import os, unittest
from unittest.mock import MagicMock, patch

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
#############################################################

import fedbiomed
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.federated_workflows import TrainingPlanWorkflow
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob, TrainingPlanApprovalJob
from testsupport.fake_training_plan import FakeTorchTrainingPlan


class TestTrainingPlanWorkflow(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests")
        super().setUp()
        self.abstract_methods_patcher = patch.multiple(TrainingPlanWorkflow, __abstractmethods__=set())
        self.abstract_methods_patcher.start()
        self.patch_job = patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingJob')
        self.mock_job = self.patch_job.start()
        mock_training_job = MagicMock(spec=TrainingJob)
        self.mock_job.return_value = mock_training_job


    def tearDown(self):
        super().tearDown()
        self.abstract_methods_patcher.stop()
        self.patch_job.stop()

    def test_training_plan_workflow_01_initialization(self):
        """Test initialization of training plan workflow, only cases where correct parameters are provided"""
        self.mock_job.return_value.get_initialized_tp_instance.return_value = MagicMock(spec=FakeTorchTrainingPlan)
        exp = TrainingPlanWorkflow()
        exp.set_training_plan_class(FakeTorchTrainingPlan)
        self.mock_job.return_value.get_initialized_tp_instance.assert_called_once()
        self.assertEqual(self.mock_job.return_value.get_initialized_tp_instance.mock_calls[0].args[0],
                         FakeTorchTrainingPlan)
        self.assertDictEqual(self.mock_job.return_value.get_initialized_tp_instance.mock_calls[0].args[1].dict(),
                             TrainingArgs({}, only_required=False).dict())
        self.assertIsNone(self.mock_job.return_value.get_initialized_tp_instance.mock_calls[0].args[2])
        # check that weights are correctly preserved
        self.mock_job.return_value.get_initialized_tp_instance.return_value.get_model_params.return_value = \
            {'model': 'params'}
        exp.set_training_plan_class(FakeTorchTrainingPlan)
        self.mock_job.return_value.get_initialized_tp_instance.return_value.set_model_params.assert_called_once_with(
            {'model': 'params'}
        )
        # check that weights are not preserved if we explicitly ask not to
        self.mock_job.return_value.get_initialized_tp_instance.return_value.set_model_params.reset_mock()
        exp.set_training_plan_class(FakeTorchTrainingPlan, keep_weights=False)
        self.assertEqual(
            self.mock_job.return_value.get_initialized_tp_instance.return_value.set_model_params.call_count, 0
        )
        # resetting training plan class to None
        exp.set_training_plan_class(None)
        self.assertIsNone(exp.training_plan_class())
        self.assertIsNone(exp.training_plan())
        # model arguments
        exp.set_training_plan_class(FakeTorchTrainingPlan)
        self.mock_job.return_value.reset_mock()
        exp.set_model_args({'model': 'args'}, keep_weights=False)
        self.assertDictEqual(exp.model_args(), {'model': 'args'})
        self.assertEqual(self.mock_job.return_value.get_initialized_tp_instance.mock_calls[0].args[0],
                         FakeTorchTrainingPlan)
        self.assertDictEqual(self.mock_job.return_value.get_initialized_tp_instance.mock_calls[0].args[1].dict(),
                             TrainingArgs({}, only_required=False).dict())
        self.assertDictEqual(self.mock_job.return_value.get_initialized_tp_instance.mock_calls[0].args[2],
                             {'model': 'args'})
        # try to keep weights
        self.mock_job.return_value.reset_mock()
        self.mock_job.return_value.get_initialized_tp_instance.return_value.set_model_params.reset_mock()
        self.mock_job.return_value.get_initialized_tp_instance.return_value.get_model_params.return_value = \
            {'model': 'new-params'}
        exp.set_model_args({'model': 'other-args'})
        self.assertDictEqual(exp.model_args(), {'model': 'other-args'})
        self.mock_job.return_value.get_initialized_tp_instance.return_value.set_model_params.assert_called_once_with(
            {'model': 'new-params'}
        )

        # TrainingPlanWorkflow can also be constructed by providing parameters to the constructor
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _training_data.node_ids.return_value = ['alice', 'bob']  # make sure that nodes can be correctly inferred
        exp = TrainingPlanWorkflow(
            nodes=['alice', 'bob'],
            training_data=_training_data,
            training_args={'num_updates': 1},
            secagg=True,
            save_breakpoints=True,
            training_plan_class=FakeTorchTrainingPlan,
            model_args={'model-args': 'from-constructor'}
        )
        self.assertDictEqual(exp.model_args(), {'model-args': 'from-constructor'})
        self.assertIsInstance(exp.training_plan(), FakeTorchTrainingPlan)
        # arguments only relevant to TrainingPlanWorkflow
        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan,
            model_args={'model-args': 'from-constructor'}
        )
        self.assertDictEqual(exp.model_args(), {'model-args': 'from-constructor'})
        self.assertIsInstance(exp.training_plan(), FakeTorchTrainingPlan)

    def test_training_plan_workflow_02_approval_and_status(self):
        """"""
        patch_job = patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingPlanApprovalJob')
        mock_job = patch_job.start()
        mock_approval_job = MagicMock(spec=TrainingPlanApprovalJob)
        mock_job.return_value = mock_approval_job
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)

        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan,
            training_data=_training_data
        )

        status = exp.check_training_plan_status()
        mock_approval_job.check_training_plan_is_approved_by_nodes.assert_called_once_with(
            job_id=exp.id,
            training_plan=exp.training_plan()
        )
        response = exp.training_plan_approve(description='some description')
        mock_approval_job.training_plan_approve.assert_called_once_with(
            training_plan=exp.training_plan(),
            description='some description'
        )


    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.FederatedWorkflow.breakpoint')
    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.uuid.uuid4', return_value='UUID')
    def test_federated_workflow_04_breakpoint(self,
                                              mock_uuid,
                                              mock_super_breakpoint,
                                              ):
        # define attributes that will be saved in breakpoint
        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlan,
            model_args={'breakpoint-model': 'args'}
        )
        exp.breakpoint(state={}, bkpt_number=1)
        # This also validates the breakpoint scheme: if this fails, please consider updating the breakpoints version
        mock_super_breakpoint.assert_called_once_with(
            {
                'model_args': {'breakpoint-model': 'args'},
                'training_plan_class_name': 'FakeTorchTrainingPlan',
                'training_plan_path': os.path.join(environ['EXPERIMENTS_DIR'],
                                                   exp.experimentation_folder(),
                                                   'breakpoint_0000',
                                                   'model_0000.py'
                                                   ),
            },
            1
        )

    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.import_class_from_file',
           return_value=(None, FakeTorchTrainingPlan))
    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.FederatedWorkflow.load_breakpoint')
    def test_federated_workflow_05_load_breakpoint(self,
                                                   mock_super_load,
                                                   mock_import_class
                                                   ):
        self.mock_job.return_value.get_initialized_tp_instance.return_value = MagicMock(spec=FakeTorchTrainingPlan)
        mock_super_load.return_value = (
            TrainingPlanWorkflow(),
            {
                'model_args': {'breakpoint-model': 'args'},
                'training_plan_class_name': 'FakeTorchTrainingPlan',
                'training_plan_path': 'some-path'
            }
        )

        exp, saved_state = TrainingPlanWorkflow.load_breakpoint()
        self.assertEqual(exp.training_plan_class(), FakeTorchTrainingPlan)
        self.assertIsInstance(exp.training_plan(), FakeTorchTrainingPlan)
        self.assertDictEqual(exp.model_args(), {'breakpoint-model': 'args'})


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
