from collections import defaultdict
import copy
import os
import shutil
import unittest
import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, call, create_autospec, patch

import numpy as np
import torch
import fedbiomed
from fedbiomed.common.exceptions import FedbiomedNodeStateAgentError
from fedbiomed.common.serializer import Serializer
from fedbiomed.researcher.datasets import FederatedDataSet

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule
#############################################################

from testsupport.fake_training_plan import FakeModel, FakeTorchTrainingPlan
from testsupport.fake_message import FakeMessages
from testsupport.fake_responses import FakeResponses
from testsupport.fake_uuid import FakeUuid
from testsupport import fake_training_plan


from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.message import TrainingPlanStatusReply, TrainingPlanStatusRequest, TrainReply, ErrorMessage
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.exceptions import FedbiomedJobError
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
import fedbiomed.researcher.job # needed for specific mocking


training_args_for_testing = TrainingArgs({"loader_args": {"batch_size": 12}}, only_required=False)


class TestJob(ResearcherTestCase, MockRequestModule):

    @classmethod
    def create_fake_model(cls, name: str):
        """ Class method saving codes of FakeModel

        Args:
            name (str): Name of the model file that will be created
        """

        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        tmp_dir_model = os.path.join(tmp_dir, name)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        content = "from testsupport.fake_training_plan import FakeModel"
        with open(tmp_dir_model, "w", encoding="utf-8") as file:
            file.write(content)

        return tmp_dir_model

    # once in test lifetime
    @classmethod
    def setUpClass(cls):

        super().setUpClass()

        def msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeMessages(msg)
            return fake_node_msg

        cls.msg_side_effect = msg_side_effect

        def fake_responses(data: Dict):
            fake = FakeResponses(data)
            return fake

        cls.fake_responses_side_effect = fake_responses


    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.job.Requests")

        self.patcher4 = patch('fedbiomed.common.message.ResearcherMessages.format_outgoing_message')
        self.patcher5 = patch('fedbiomed.researcher.job.atexit')
        self.ic_from_file_patch = patch('fedbiomed.researcher.job.utils.import_class_object_from_file', autospec=True)
        
        self.mock_request_create = self.patcher4.start()
        self.mock_atexit = self.patcher5.start()
        self.ic_from_file_mock = self.ic_from_file_patch.start()

        # Globally create mock for Model and FederatedDataset
        self.model = create_autospec(BaseTrainingPlan, instance=False)
        self.fds = MagicMock(spec=FederatedDataSet)
        self.fds.data = MagicMock(return_value={})

        self.mock_request_create.side_effect = TestJob.msg_side_effect

        self.model = FakeTorchTrainingPlan
        self.model.save_code = MagicMock()
        
        self.ic_from_file_mock.return_value = (fake_training_plan, FakeTorchTrainingPlan() )
        # Build Global Job that will be used in most of the tests
        self.job = Job(
            training_plan_class=self.model,
            training_args=training_args_for_testing,
            data=self.fds
        )

    def tearDown(self) -> None:

        self.patcher4.stop()
        self.patcher5.stop()
        self.ic_from_file_patch.stop()
        # shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
        # (above) remove files created during these unit tests

        # Remove if there is dummy model file
        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

        super().tearDown()


    @patch('fedbiomed.common.logger.logger.critical')
    def test_job_01_init_t1(self,
                            mock_logger_critical):
        """ Test first raise error when there is no model provided """
        mock_logger_critical.return_value = None

        with self.assertRaises(FedbiomedJobError):
            _ = Job()
            mock_logger_critical.assert_called_once()

    def test_job_02_init_keep_files_dir(self):
        """ Testing initialization of Job with keep_files_dir """

        j = Job(training_plan_class=self.model,
                data=self.fds,
                training_args=training_args_for_testing,
                keep_files_dir=environ['TMP_DIR'])

        # Check keep files dir properly set
        self.assertEqual(j._keep_files_dir, environ['TMP_DIR'], 'keep_files_dir does not matched given path')

    def test_job_03_init_provide_request(self):
        """ Testing initialization of Job by providing Request object """

        reqs = MagicMock(spec=Requests)
        j = Job(training_plan_class=self.model,
                training_args=training_args_for_testing,
                data=self.fds,
                reqs=reqs)

        self.assertEqual(j._reqs, reqs, 'Job did not initialize provided Request object')


    def test_job_06_init_isclass_raises_error(self):
        """ Test initialization when inspect.isclass raises NameError"""

        with self.assertRaises(FedbiomedJobError):
            _ = Job(training_plan_class='FakeModel',
                    training_args=training_args_for_testing,
                    data=self.fds)

        with self.assertRaises(FedbiomedJobError):
            _ = Job(training_plan_class=FakeModel,
                    training_args=training_args_for_testing,
                    data=self.fds)

    def test_job_07_initialization_raising_exception_save_and_save_code(self):

        """ Test Job initialization when model_instance.save and save_code raises Exception """


        # Test TRY/EXCEPT when save_code raises Exception
        self.model.save_code.side_effect = Exception
        with self.assertRaises(FedbiomedJobError):
            _ = Job(training_plan_class=self.model,
                    training_args=training_args_for_testing,
                    data=self.fds)


    def test_job_08_properties_setters(self):
        """ Testing all properties and setters of Job class
            TODO: Change this part after refactoring getters and setters
        """
        self.assertEqual(self.model.__name__, self.job.training_plan.__class__.__name__)
        self.assertEqual(self.job._reqs, self.job.requests, 'Can not get Requests attribute from Job properly')

        nodes = {'node-1': 1, 'node-2': 2}
        self.job.nodes = nodes
        self.assertDictEqual(nodes, self.job.nodes, 'Can not set or get properly nodes attribute of Job')

        tr = self.job.training_replies
        self.assertEqual(self.job._training_replies, tr, 'Can not get training_replies correctly')

        self.job.training_args = TrainingArgs({'loader_args': {'batch_size': 33}})
        targs = self.job.training_args
        self.assertEqual(33, targs['loader_args']['batch_size'], 'Can not get or set training_args correctly')


    def test_job_09_check_training_plan_is_approved_by_nodes(self):
        """ Testing the method that check training plan approval status of the nodes"""

        self.fds.node_ids = MagicMock(return_value=['node-1', 'node-2'])
        base_reply = {'researcher_id': self.job._researcher_id, "job_id": self.job._id, 'msg':'test', 'training_plan': 'test', 'command': 'training-plan-status'}

        # Test when model is approved by all nodes
        responses = {
            'node-1': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-1', 'success': True, 'approval_obligation': True, 'status': 'APPROVED'}),
            'node-2': TrainingPlanStatusReply(**{**base_reply,'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'status': 'APPROVED'})
        } 

        self.mock_federated_request.replies.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()

        self.assertDictEqual(responses, result,
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when model is approved by only one node
        responses = {
            'node-1': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-1', 'success': True, 'approval_obligation': True, 'status': 'APPROVED'}),
            'node-2': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'status': 'REJECTED'})
        } 
        self.mock_federated_request.replies.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertDictEqual(responses, result,
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when training plan approval obligation is False by one node
        responses = {
            'node-1': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-1', 'success': True, 'approval_obligation': False, 'status': 'REJECTED'}),
            'node-2': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-2', 'success': True, 'approval_obligation': True,  'status': 'APPROVED'})
        } 
        self.mock_federated_request.replies.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertDictEqual(responses, result,
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when one of the reply success status is False
        responses = {
            'node-1': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-1', 'success': False, 'approval_obligation': False, 'status': 'REJECTED'}),
            'node-2': TrainingPlanStatusReply(**{**base_reply, 'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'status': 'REJECTED'})
        } 
        self.mock_federated_request.replies.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertDictEqual(responses, result,
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when one of the nodes does not reply
        responses = {
            'node-1': TrainingPlanStatusReply(**{**base_reply,'node_id': 'node-1', 'success': True, 'approval_obligation': False, 'status': 'REJECTED'})
        } 
        self.mock_federated_request.replies.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertDictEqual(responses, result,
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

    def test_job_11_start_training_round(self):
        """ Test Job - start_training_round method with 3 different scenarios

            Test - 1 : When all the nodes successful completes training
            Test - 2 : When one of the nodes returns error during training
            Test - 3 : When one of the nodes returns error without extra_msg
                       This test also checks whether previous node (which returned)
                       error is removed or not
        """
        train_reply_base = {'job_id': self.job._id, 'researcher_id': environ['RESEARCHER_ID']}
        self.job._nodes = ['node-1', 'node-2']
        self.job._model_args = {}
        self.fds.data = MagicMock(return_value={
            'node-1': {'dataset_id': '1234'},
            'node-2': {'dataset_id': '12345'}
        })

        reply_base = TrainReply(**{
            'node_id': 'node-1', 'researcher_id': environ['RESEARCHER_ID'],
            'job_id': self.job._id,
            'state_id': 'node_state_id_1234',
            'params': {"x": 0},
            'optimizer_args': None,
            'optim_aux_var': None,
            'encryption_factor': None,
            'timing': {'rtime_total': 12},
            'success': True,
            'msg': 'MSG',
            'dataset_id': '1234',
            'command': 'train',
            'sample_size': 100})
        replies = {
            'node-1': reply_base,
            'node-2': reply_base
        }

        replies['node-1'].node_id = 'node-1'
        replies['node-2'].node_id = 'node-2'
        replies['node-2'].state_id = 'node_state_id_4321'


        reply_error_1 = ErrorMessage(**{
            'node_id': 'node-2', 'researcher_id': environ['RESEARCHER_ID'],
            'errnum': ErrorNumbers.FB100.value,
            'extra_msg': 'this extra msg',
            'command': 'error'})

        reply_error_2 = ErrorMessage(**{
            'node_id': 'node-2', 'researcher_id': environ['RESEARCHER_ID'],
            'extra_msg': '',
            'errnum': ErrorNumbers.FB100.value,
            'command': 'error'})


        self.mock_federated_request.replies.return_value = replies
        aggregator_args = { node_id: {'aggregator_name': 'my_aggregator'} for node_id in self.job._nodes}

        # Test - 1
        nodes = self.job.start_nodes_training_round(1, aggregator_args=aggregator_args)
        self.mock_requests.return_value.send.assert_called_once()
        self.assertListEqual(nodes, ['node-1', 'node-2'])

        # Test - 2 When one of the nodes returns error
        self.mock_federated_request.replies.return_value = {'node-1': replies['node-1']}
        self.mock_federated_request.errors.return_value  = {'node-2': reply_error_1}
        nodes = self.job.start_nodes_training_round(2, aggregator_args=aggregator_args)
        self.assertListEqual(nodes, ['node-1'])

        # Test - 2 When one of the nodes returns error without extra_msg and
        # check node-2 is removed since it returned error in the previous test call
        self.job._nodes = ['node-1', 'node-2']
        self.mock_federated_request.replies.return_value = {'node-1': replies['node-1']}
        self.mock_federated_request.errors.return_value  = {'node-2': reply_error_2}
        nodes = self.job.start_nodes_training_round(3, aggregator_args=aggregator_args)
        self.assertListEqual(nodes, ['node-1'])

    def test_job_12_start_nodes_training_round_optim_aux_var(self):
        """Test that 'optim_aux_var' is properly used in 'start_nodes_training_round'."""
        fake_aux_var = {"module": {"key": "val"}}
        # General setup: skip requests sending and replies processing.
        self.job.nodes = []

        self.job.start_nodes_training_round(
            1, {}, {}, do_training=True, optim_aux_var=fake_aux_var
        )

        self.job.start_nodes_training_round(
                1, {}, {}, do_training=False, optim_aux_var=fake_aux_var
        )

    def test_job_14_update_parameters_from_params(self):
        """Testing update_parameters when passing 'params'."""
        params = {'params': [1, 2, 3, 4]}
        with (
            patch("fedbiomed.common.serializer.Serializer.dump") as patch_dump
        ):
            result = self.job.update_parameters(params=params)
            self.assertEqual(self.job._model_params_file, result)


    def test_job_20_save_private_training_replies(self):
        """
        tests if `_save_training_replies` is properly extracting
        breakpoint info from `training_replies`. It uses a dummy class
        FakeResponses, a weak implementation of `Responses` class
        """

        # first create a `_training_replies` variable
        training_replies = { 0: {
            'node-1': {
                "node_id": '1234',
                'state_id': '1234',
                'params': torch.Tensor([1, 3, 5]),
                'dataset_id': 'id_node_1'},
            'node-2': {
                "node_id": '5678',
                'state_id': '5678',
                'params': np.array([1, 3, 5]),
                'dataset_id': 'id_node_2'}}}

        # action
        new_training_replies = self.job._save_training_replies(training_replies)

        # check if `training_replies` is  saved accordingly
        self.assertTrue(type(new_training_replies) is list)
        self.assertTrue(len(new_training_replies) == 1)
        self.assertTrue('params' not in new_training_replies[0]['node-1'])
        self.assertEqual(new_training_replies[0]['node-2'].get('dataset_id'), 'id_node_2')


    def test_job_21_private_load_training_replies(self):
        """tests if `_load_training_replies` is loading file content from path file
        and is building a proper training replies structure from breakpoint info
        """
        # Declare mock model parameters, for torch and scikit-learn.
        pytorch_params = torch.Tensor([1, 3, 5, 7])
        sklearn_params = np.array([[1, 2, 3, 4, 5], [2, 8, 7, 5, 5]])

        # mock FederatedDataSet
        fds = MagicMock(spec=FederatedDataSet)
        fds.data = MagicMock(spec=dict, return_value={})


        # instantiate job with a mock training plan
        test_job_torch = Job(
            training_plan_class=FakeTorchTrainingPlan,
            training_args=training_args_for_testing,
            data=fds
        )
        # second create a `training_replies` variable
        loaded_training_replies_torch = [
            
                {
                    "node_1234": {"success": True,
                        "msg": "",
                        "dataset_id": "dataset_1234",
                        "node_id": "node_1234",
                        "state_id": "state_1234",
                        "params_path": "/path/to/file/param.mpk",
                        "timing": {"time": 0}},
                    "node_4567": {"success": True,
                        "msg": "",
                        "dataset_id": "dataset_4567",
                        "node_id": "node_4567",
                        "state_id": "state_4567",
                        "params_path": "/path/to/file/param2.mpk",
                        "timing": {"time": 0}}
                }
            
        ]

        # action
        with patch(
            "fedbiomed.common.serializer.Serializer.load", return_value=pytorch_params
        ) as load_patch:
            torch_training_replies = test_job_torch._load_training_replies(
                loaded_training_replies_torch
            )
        self.assertEqual(load_patch.call_count, 2)
        load_patch.assert_called_with(
            loaded_training_replies_torch[0]['node_4567']["params_path"],
        )
        self.assertIsInstance(torch_training_replies, dict)
        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            torch_training_replies[0]['node_1234']['node_id'],
            loaded_training_replies_torch[0]['node_1234']['node_id'])
        # check `training_replies` for pytorch models
        self.assertTrue(torch.eq(
            torch_training_replies[0]['node_4567']['params'],
            pytorch_params
        ).all())
        self.assertEqual(
            torch_training_replies[0]['node_4567']['params_path'],
            "/path/to/file/param2.mpk"
        )
        self.assertTrue(isinstance(torch_training_replies[0], Dict))

      
    @patch('fedbiomed.researcher.job.Job._load_training_replies')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    @patch('fedbiomed.researcher.job.Serializer.load')
    def test_job_22_load_state_breakpoint(
            self,
            patch_job_update_parameters,
            patch_job_load_training_replies,
            load_mock
    ):
        """
        test if the job state values correctly initialize job (from breakpoints)
        """
        node_states = {
            'collection_state_ids': {'node_id_xxx': 'state_id_xxx', 'node_id_xyx': 'state_id_xyx'},
        }
        
        # modifying FederatedDataset mock 
        self.fds.data.return_value = node_states
        self.job.nodes = node_states
        job_state = {
            'researcher_id': 'my_researcher_id_123456789',
            'job_id': 'my_job_id_abcdefghij',
            'model_params_path': '/path/to/my/model_file.py',
            'training_replies': {0: 'un', 1: 'deux'},
            'node_state': node_states
        }
        new_training_replies = {2: 'trois', 3: 'quatre'}

        # patch `update_parameters`
        patch_job_update_parameters.return_value = "dummy_string"

        # patch `_load_training_replies`
        patch_job_load_training_replies.return_value = new_training_replies
        load_mock.return_value = new_training_replies
        # action
        self.job.load_state_breakpoint(job_state)

        self.assertEqual(self.job._researcher_id, job_state['researcher_id'])
        self.assertEqual(self.job._id, job_state['job_id'])
        self.assertEqual(self.job._training_replies, new_training_replies)
        self.assertDictEqual(self.job._node_state_agent.get_last_node_states(), node_states['collection_state_ids'])

    @patch('fedbiomed.researcher.job.create_unique_link')
    @patch('fedbiomed.researcher.job.create_unique_file_link')
    @patch('fedbiomed.researcher.job.Job._save_training_replies')
    def test_job_23_save_state_breakpoint(
            self,
            patch_job_save_training_replies,
            patch_create_unique_file_link,
            patch_create_unique_link
    ):
        """
        test that job breakpoint state structure + file links are created
        """
        fds_content = {'node_id_xxx': MagicMock(), 'node_id_xyx': MagicMock()}
        
        # modifying FederatedDataset mock 
        self.fds.data.return_value = fds_content
        
        test_job = Job(
            training_plan_class=self.model,
            training_args=training_args_for_testing,
            data=self.fds
        )

        new_training_replies = [{
            "node-1": {'params_path': '/path/to/job_test_save_state_params0.pt'},
            "node-2": {'params_path': '/path/to/job_test_save_state_params1.pt'},
        }]
        # expected transformed values of new_training_replies for save state
        new_training_replies_state = [{
            "node-1": {'params_path': 'xxx/job_test_save_state_params0.pt'},
            "node-2": {'params_path': 'xxx/job_test_save_state_params1.pt'},
        }]
        

        link_path = '/path/to/job_test_save_state_params_link.pt'

        # patches configuration
        patch_create_unique_link.return_value = link_path

        def side_create_ufl(breakpoint_folder_path, file_path):
            return os.path.join(breakpoint_folder_path, os.path.basename(file_path))

        patch_create_unique_file_link.side_effect = side_create_ufl

        patch_job_save_training_replies.return_value = new_training_replies

        # choose arguments for saving state
        breakpoint_path = 'xxx'

        # action
        save_state_bkpt = test_job.save_state_breakpoint(breakpoint_path)
        node_states = {k: None for k in fds_content}
        self.assertEqual(environ['RESEARCHER_ID'], save_state_bkpt['researcher_id'])
        self.assertEqual(test_job._id, save_state_bkpt['job_id'])
        self.assertEqual(link_path, save_state_bkpt['model_params_path'])
        self.assertDictEqual(node_states, save_state_bkpt['node_state']['collection_state_ids'])


    def test_job_25_extract_received_optimizer_aux_var_from_round(self):
        """Test that 'extract_received_optimizer_aux_var_from_round' works well."""
        # Set up: nodes sent back some Optimizer aux var information.
        responses = {}
        responses.update({'node-1': {
            "node_id": "node-1",
            "optim_aux_var": {
                "module_a": {"key": "a1"}, "module_b": {"key": "b1"}
            }}})

        responses.update({'node-2': {
            "node_id": "node-2",
            "optim_aux_var": {
                "module_a": {"key": "a2"}, "module_b": {"key": "b2"}
            }}})

        getattr(self.job, "_training_replies")[1] = responses
        # Call the method and verify that its output matches expectations.
        aux_var = self.job.extract_received_optimizer_aux_var_from_round(round_id=1)
        expected = {
            "module_a": {"node-1": {"key": "a1"}, "node-2": {"key": "a2"}},
            "module_b": {"node-1": {"key": "b1"}, "node-2": {"key": "b2"}},
        }
        self.assertDictEqual(aux_var, expected)

    def test_job_26_extract_received_optimizer_aux_var_from_round_empty(self):
        """Test 'extract_received_optimizer_aux_var_from_round' without aux var."""
        # Set up: nodes did not send Optimizer aux var information.
        responses = {}
        responses.update({'node-1': {
            "node_id": "node-1"}})

        responses.update({'node-2': {
            "node_id": "node-2"}})

        getattr(self.job, "_training_replies")[1] = responses
        # Call the method and verify that it returns an empty dict.
        aux_var = self.job.extract_received_optimizer_aux_var_from_round(round_id=1)
        self.assertDictEqual(aux_var, {})


    @patch('fedbiomed.researcher.job.create_unique_link')
    @patch('fedbiomed.researcher.job.create_unique_file_link')
    def test_job_28_update_node_state_agent(self,
                                            patch_create_unique_file_link,
                                            patch_create_unique_link):
        # FIXME: this is more like an integration test rather than a unit test
        # it should be defined in a specific class that contains all integration tests
        # modifying fds of Job
        
        def set_training_replies_through_bkpt(job: Job, states: Dict):
            """Sets Job's `training_replies` private attributethrough breakpoint API

            Args:
                job (Job): job object to be updated
                states (Dict): job_state
            """
            with (patch('fedbiomed.researcher.job.Job.update_parameters') as update_param_patch,
                  patch.object( Serializer, 'load') as serializer_patch):
                serializer_patch = MagicMock(return_value={'model_weights': 1234})
                job.load_state_breakpoint(states)

        data = {
            'node-1': [{'dataset_id': 'dataset-id-1',
                        'shape': [100, 100]}],
            'node-2': [{'dataset_id': 'dataset-id-2',
                        'shape': [120, 120], 
                        'test_ratio': .0}],
            'node-3': [{'dataset_id': 'dataset-id-3',
                        'shape': [120, 120], 
                        'test_ratio': .0}],
            'node-4': [{'dataset_id': 'dataset-id-4',
                        'shape': [120, 120], 
                        'test_ratio': .0}],
        }
        fds = FederatedDataSet(data)

        loaded_training_replies_torch = [{ 
            'node-1': {
                "success": True,
                "msg": "",
                "dataset_id": "dataset_1234",
                "node_id": "node-3",
                "state_id": "state_1234",
                "params_path": "/path/to/file/param.mpk",
                "timing": {"time": 0}},
            'node-2': {
                "success": True,
                "msg": "",
                "dataset_id": "dataset_4567",
                "node_id": "node-4",
                "state_id": "state_4567",
                "params_path": "/path/to/file/param2.mpk",
                "timing": {"time": 0}}}
        ]

        test_job = Job(
            training_plan_class=self.model,
            training_args=training_args_for_testing,
            data=fds)

        # do test while `before_training` = True
        test_job._update_nodes_states_agent(before_training=True)
        # action! collect Job state in order to retrieve state_ids

        job_state = test_job.save_state_breakpoint("path/to/bkpt")
        self.assertDictEqual({k: None for k in data.keys()}, job_state['node_state']['collection_state_ids'])

        # we cannot access to `_training_replies` private attribute hence the saving/loading part
        job_state.update({'training_replies': loaded_training_replies_torch})  

        set_training_replies_through_bkpt(test_job, job_state)

        # action! 
        test_job._update_nodes_states_agent(before_training=False)

        # retrieving NodeAgentState through Job state
        job_state = test_job.save_state_breakpoint("path/to/bkpt/2")

        # check that NodeStateAgent have been updated accordingly
        for node_id, reply in loaded_training_replies_torch[-1].items():
            self.assertIn(node_id, list(job_state['node_state']['collection_state_ids']))

        # finally we are testing that exception is raised if index cannot be extracted

        loaded_training_replies = []
        job_state.update({'training_replies': loaded_training_replies})  

        set_training_replies_through_bkpt(test_job, job_state)

        with self.assertRaises(FedbiomedNodeStateAgentError):
            test_job._update_nodes_states_agent(before_training=False)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
