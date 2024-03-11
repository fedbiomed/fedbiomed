import os, unittest
from unittest.mock import MagicMock, patch
from itertools import product


#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
#############################################################

import fedbiomed
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.federated_workflows import TrainingPlanWorkflow
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob, TrainingPlanApprovalJob
from testsupport.fake_training_plan import (
    FakeTorchTrainingPlan,
    FakeTorchTrainingPlanForClassSource,
    FakeSKLearnTrainingPlanForClassSource
)

class TestTrainingPlanWorkflow(ResearcherTestCase, MockRequestModule):

    def setUp(self):
        MockRequestModule.setUp(
            self,
            module="fedbiomed.researcher.federated_workflows._federated_workflow.Requests"
        )

        super().setUp()
        self.abstract_methods_patcher = patch.multiple(
            TrainingPlanWorkflow, __abstractmethods__=set()
        )
        self.abstract_methods_patcher.start()
        self.patch_job = patch(
            'fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingJob'
        )
        self.patch_ttp_post_init = patch(
            'fedbiomed.common.training_plans.TorchTrainingPlan.post_init')

        self.patch_stp_post_init = patch(
             'fedbiomed.common.training_plans.SKLearnTrainingPlan.post_init')

        self.mock_ttp_post_init = self.patch_ttp_post_init.start()
        self.mock_stp_post_init = self.patch_stp_post_init.start()

        self.mock_job = self.patch_job.start()
        mock_training_job = MagicMock(spec=TrainingJob)
        self.mock_job.return_value = mock_training_job


    def tearDown(self):
        super().tearDown()
        self.abstract_methods_patcher.stop()
        self.patch_job.stop()

    def test_training_plan_workflow_01_initialization(self):
        """Test initialization of training plan workflow, only cases where correct parameters are provided"""
        #self.mock_job.return_value.get_initialized_tp_instance.return_value = \
        #    MagicMock(spec=FakeTorchTrainingPlan)

        exp = TrainingPlanWorkflow()
        self.assertIsNone(exp.training_plan_class())
        self.assertIsNone(exp.model_args())
        self.assertIsNone(exp.training_plan())


        # Test all possible combinations of init arguments
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _secagg = MagicMock(spec=fedbiomed.researcher.secagg.SecureAggregation)
        parameters_and_possible_values = {
            'tags': ('one-tag', ['one-tag', 'another-tag'], None),
            'nodes': (['one-node'], None),
            'training_data': (_training_data, {'one-node': {'tags': ['one-tag']}}, None),
            'training_args': (TrainingArgs({'epochs': 42}), {'num_updates': 1}, None),
            'experimentation_folder': ('folder_name', None),
            'secagg': (True, False, _secagg),
            'save_breakpoints': (True, False),
            #'training_plan_class': (TorchTrainingPlan, SKLearnTrainingPlan, None),
            'training_plan_class': (
                FakeTorchTrainingPlanForClassSource, FakeSKLearnTrainingPlanForClassSource, None),
            'model_args': ({'model': 'args'}, None)
        }
        # Compute cartesian product of parameter values to obtain all possible combinations
        keys, values = zip(*parameters_and_possible_values.items())
        all_parameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
        for params in all_parameter_combinations:
            try:
                exp = TrainingPlanWorkflow(**params)
            except Exception as e:
                print(f'Exception {e} raised with the following parameters {params}')
                raise e

        # Special corner cases that deserve additional testing
        # TrainingPlanWorkflow can also be constructed by providing parameters to the constructor
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)
        _training_data.node_ids.return_value = ['alice', 'bob']  # make sure that nodes can be correctly inferred
        exp = TrainingPlanWorkflow(
            nodes=['alice', 'bob'],
            training_data=_training_data,
            training_args={'num_updates': 1},
            secagg=True,
            save_breakpoints=True,
            training_plan_class=FakeTorchTrainingPlanForClassSource,
            model_args={'model-args': 'from-constructor'}
        )
        self.assertDictEqual(exp.model_args(), {'model-args': 'from-constructor'})
        self.assertTrue(
            exp.training_plan().__class__.__name__ == \
            FakeTorchTrainingPlanForClassSource.__name__)

        # arguments only relevant to TrainingPlanWorkflow
        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlanForClassSource,
            model_args={'model-args': 'from-constructor'}
        )
        self.assertDictEqual(exp.model_args(), {'model-args': 'from-constructor'})
        self.assertTrue(
            exp.training_plan().__class__.__name__ == \
            FakeTorchTrainingPlanForClassSource.__name__)

    def test_training_plan_workflow_02_set_training_plan_class(self):
        """Tests setting training plan class"""
        exp = TrainingPlanWorkflow()

        # Check if correct training plan instantiated
        exp.set_training_plan_class(FakeTorchTrainingPlanForClassSource)
        self.assertIsInstance(exp.training_plan(), TorchTrainingPlan)

        # Following set complains about get_weights. It is becaise post_init method of
        # SkLearnTrainingPlan is mocked. Therefore, self._model is None
        # exp.set_training_plan_class(FakeSKLearnTrainingPlanForClassSource)
        # self.assertIsInstance(exp.training_plan(), SKLearnTrainingPlan)

        # check that weights are not preserved if we explicitly ask not to
        # resetting training plan class to None
        exp.set_training_plan_class(None)
        self.assertIsNone(exp.training_plan_class())
        self.assertIsNone(exp.training_plan())

    def test_training_plan_workflow_03_set_model_args(self):
        """Tests set model args"""

        exp = TrainingPlanWorkflow()
        exp.set_training_plan_class(FakeTorchTrainingPlanForClassSource)

        self.mock_ttp_post_init.reset_mock()
        exp.set_model_args({'model': 'args'}, keep_weights=False)
        self.mock_ttp_post_init.assert_called_once()


    def test_training_plan_workflow_04_approval_and_status(self):
        """"""
        patch_job = patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.TrainingPlanApprovalJob')
        mock_job = patch_job.start()
        mock_approval_job = MagicMock(spec=TrainingPlanApprovalJob)
        mock_job.return_value = mock_approval_job
        _training_data = MagicMock(spec=fedbiomed.researcher.datasets.FederatedDataSet)

        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlanForClassSource,
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
    def test_federated_workflow_05_breakpoint(self,
                                              mock_uuid,
                                              mock_super_breakpoint,
                                              ):
        # define attributes that will be saved in breakpoint
        exp = TrainingPlanWorkflow(
            training_plan_class=FakeTorchTrainingPlanForClassSource,
            model_args={'breakpoint-model': 'args'}
        )
        exp.breakpoint(state={}, bkpt_number=1)
        # This also validates the breakpoint scheme: if this fails, please consider updating the breakpoints version
        mock_super_breakpoint.assert_called_once_with(
            {
                'model_args': {'breakpoint-model': 'args'},
                'training_plan_class_name': 'FakeTorchTrainingPlanForClassSource',
                'training_plan_path': os.path.join(environ['EXPERIMENTS_DIR'],
                                                   exp.experimentation_folder(),
                                                   'breakpoint_0000',
                                                   'model_0000.py'
                                                   ),
            },
            1
        )

    @patch(
        'fedbiomed.researcher.federated_workflows._training_plan_workflow.import_class_from_file',
        return_value=(None, FakeTorchTrainingPlanForClassSource))
    @patch(
    'fedbiomed.researcher.federated_workflows._training_plan_workflow.'
    'FederatedWorkflow.load_breakpoint'
    )
    def test_federated_workflow_06_load_breakpoint(self,
                                                   mock_super_load,
                                                   mock_import_class
                                                   ):
        mock_super_load.return_value = (
            TrainingPlanWorkflow(),
            {
                'model_args': {'breakpoint-model': 'args'},
                'training_plan_class_name': 'FakeTorchTrainingPlanForClassSource',
                'training_plan_path': 'some-path'
            }
        )

        exp, saved_state = TrainingPlanWorkflow.load_breakpoint()
        self.assertEqual(exp.training_plan_class(), FakeTorchTrainingPlanForClassSource)
        self.assertIsInstance(exp.training_plan(), TorchTrainingPlan)
        self.assertDictEqual(exp.model_args(), {'breakpoint-model': 'args'})

    # TODO: Add test for _keep_weights method to test

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
