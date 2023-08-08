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

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from testsupport.fake_training_plan import FakeModel
from testsupport.fake_message import FakeMessages
from testsupport.fake_responses import FakeResponses
from testsupport.fake_uuid import FakeUuid

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
import fedbiomed.researcher.job # needed for specific mocking


training_args_for_testing = TrainingArgs({"loader_args": {"batch_size": 12}}, only_required=False)


class TestJob(ResearcherTestCase):

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

        self.patcher1 = patch('fedbiomed.researcher.requests.Requests.__init__',
                              return_value=None)
        self.patcher2 = patch('fedbiomed.common.repository.Repository.upload_file',
                              return_value={"file": environ['UPLOADS_URL']})
        self.patcher3 = patch('fedbiomed.common.repository.Repository.download_file',
                              return_value=(True, environ['TMP_DIR']))
        self.patcher4 = patch('fedbiomed.common.message.ResearcherMessages.format_outgoing_message')
        self.patcher5 = patch('fedbiomed.researcher.job.atexit')


        self.mock_request = self.patcher1.start()
        self.mock_upload_file = self.patcher2.start()
        self.mock_download_file = self.patcher3.start()
        self.mock_request_create = self.patcher4.start()
        self.mock_atexit = self.patcher5.start()

        # Globally create mock for Model and FederatedDataset
        self.model = create_autospec(BaseTrainingPlan, instance=False)
        self.fds = MagicMock()

        self.fds.data = MagicMock(return_value={})
        self.mock_request_create.side_effect = TestJob.msg_side_effect

        # Build Global Job that will be used in most of the tests
        self.job = Job(
            training_plan_class=self.model,
            training_args=training_args_for_testing,
            data=self.fds
        )

    def tearDown(self) -> None:

        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()

        # shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
        # (above) remove files created during these unit tests

        # Remove if there is dummy model file
        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    @patch('fedbiomed.common.logger.logger.critical')
    def test_job_01_init_t1(self,
                            mock_logger_critical):
        """ Test first raise error when there is no model provided """
        mock_logger_critical.return_value = None

        with self.assertRaises(NameError):
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

        reqs = Requests()
        j = Job(training_plan_class=self.model,
                training_args=training_args_for_testing,
                data=self.fds,
                reqs=reqs)

        self.assertEqual(j._reqs, reqs, 'Job did not initialize provided Request object')

    def test_job_04_init_building_model_from_path(self):
        """ Test model is passed as static python file with training_plan_path """

        # Get source of the model and save in tmp directory for just test purposes
        tmp_dir_model = TestJob.create_fake_model('fake_model.py')
        self.mock_upload_file.reset_mock()

        j = Job(training_plan_path=tmp_dir_model,
                training_args=training_args_for_testing,
                training_plan_class='FakeModel')

        self.assertEqual(j.training_plan.__class__.__name__, FakeModel.__name__,
                         'Provided model and model instance of Job do not match, '
                         'while initializing Job with static model python file')

        self.assertEqual(j._training_plan_name, 'FakeModel',
                         'Model is not initialized properly while providing training_plan_path')

        # # Upload file must be called 2 times one for model
        # # another one for initial model parameters
        self.assertEqual(self.mock_upload_file.call_count, 2)

    @patch('fedbiomed.common.logger.logger.critical')
    def test_job_init_05_build_wrongly_saved_model(self, mock_logger_critical):
        """ Testing when model code saved with unsupported module name

            - This test will catch raise SystemExit
        """

        mock_logger_critical.return_value = None

        # Save model with unsupported module name
        tmp_dir_model = TestJob.create_fake_model('fake.model.py')

        with self.assertRaises(SystemExit):
            _ = Job(training_plan_path=tmp_dir_model,
                    training_args=training_args_for_testing,
                    training_plan_class='FakeModel')
            mock_logger_critical.assert_called_once()

    @patch('fedbiomed.common.logger.logger.critical')
    def test_job_06_init_isclass_raises_error(self,
                                              mock_logger_critical):
        """ Test initialization when inspect.isclass raises NameError"""

        with patch.object(fedbiomed.researcher.job, 'inspect') as mock_inspect:
            mock_inspect.isclass.side_effect = NameError
            with self.assertRaises(NameError):
                _ = Job(training_plan_class='FakeModel',
                        training_args=training_args_for_testing,
                        data=self.fds)
                mock_logger_critical.assert_called_once()

    @patch('fedbiomed.common.logger.logger.error')
    def test_job_07_initialization_raising_exception_save_and_save_code(self,
                                                                        mock_logger_error):

        """ Test Job initialization when model_instance.save and save_code raises Exception """

        mock_logger_error.return_value = None

        # Test TRY/EXCEPT when save_code raises Exception
        self.model.save_code.side_effect = Exception
        _ = Job(training_plan_class=self.model,
                training_args=training_args_for_testing,
                data=self.fds)
        mock_logger_error.assert_called_once()

        # Reset mocks for next tests
        self.model.save_code.side_effect = None
        mock_logger_error.reset_mock()

        # Test TRY/EXCEPT when model.save() raises Exception
        self.model.get_model_params.side_effect = Exception
        _ = Job(training_plan_class=self.model,
                training_args=training_args_for_testing,
                data=self.fds)
        mock_logger_error.assert_called_once()

    def test_job_08_properties_setters(self):
        """ Testing all properties and setters of Job class
            TODO: Change this part after refactoring getters and setters
        """
        self.assertEqual(self.model, self.job.training_plan,
                         'Can not get Requests attribute from Job properly')
        self.assertEqual('BaseTrainingPlan', self.job.training_plan_name, 'Can not model class properly')
        self.assertEqual(self.job._reqs, self.job.requests, 'Can not get Requests attribute from Job properly')

        model_file = self.job.training_plan_file
        self.assertEqual(model_file, self.job.training_plan_file, 'model_file attribute of job is not got correctly')

        nodes = {'node-1': 1, 'node-2': 2}
        self.job.nodes = nodes
        self.assertDictEqual(nodes, self.job.nodes, 'Can not set or get properly nodes attribute of Job')

        tr = self.job.training_replies
        self.assertEqual(self.job._training_replies, tr, 'Can not get training_replies correctly')

        self.job.training_args = TrainingArgs({'loader_args': {'batch_size': 33}})
        targs = self.job.training_args
        self.assertEqual(33, targs['loader_args']['batch_size'], 'Can not get or set training_args correctly')

    @patch('fedbiomed.researcher.requests.Requests.send_message')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_job_09_check_training_plan_is_approved_by_nodes(self,
                                                             mock_requests_get_responses,
                                                             mock_requests_send_message):
        """ Testing the method that check training plan approval status of the nodes"""

        self.fds.node_ids = MagicMock(return_value=['node-1', 'node-2'])
        mock_requests_send_message.return_value = None

        message = {'researcher_id': self.job._researcher_id,
                   'job_id': self.job._id,
                   'training_plan_url': self.job._repository_args['training_plan_url'],
                   'command': 'training-plan-status'}

        # Test when model is approved by all nodes
        responses = FakeResponses(
            [
                {'node_id': 'node-1', 'success': True, 'approval_obligation': True, 'is_approved': True},
                {'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'is_approved': True}
            ]
        )
        mock_requests_get_responses.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        calls = mock_requests_send_message.call_args_list
        self.assertListEqual(list(calls[0][0]), [message, 'node-1'])
        self.assertListEqual(list(calls[1][0]), [message, 'node-2'])

        self.assertListEqual(responses.data(), result.data(),
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when model is approved by only one node
        responses = FakeResponses([
            {'node_id': 'node-1', 'success': True, 'approval_obligation': True, 'is_approved': True},
            {'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'is_approved': False}
        ])
        mock_requests_get_responses.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertListEqual(responses.data(), result.data(),
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when training plan approval obligation is False by one node
        responses = FakeResponses([
            {'node_id': 'node-1', 'success': True, 'approval_obligation': False, 'is_approved': False},
            {'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'is_approved': True}
        ])
        mock_requests_get_responses.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertListEqual(responses.data(), result.data(),
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when one of the reply success status is False
        responses = FakeResponses([
            {'node_id': 'node-1', 'success': False, 'approval_obligation': False, 'is_approved': False},
            {'node_id': 'node-2', 'success': True, 'approval_obligation': True, 'is_approved': True}
        ])
        mock_requests_get_responses.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertListEqual(responses.data(), result.data(),
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

        # Test when one of the nodes does not reply
        responses = FakeResponses([
            {'node_id': 'node-1', 'success': True, 'approval_obligation': False, 'is_approved': False}
        ])
        mock_requests_get_responses.return_value = responses
        result = self.job.check_training_plan_is_approved_by_nodes()
        self.assertListEqual(list(calls[0][0]), [message, 'node-1'])
        self.assertListEqual(list(calls[1][0]), [message, 'node-2'])
        self.assertListEqual(responses.data(), result.data(),
                             'Response of `check_training_plan_is_approved_by_nodes` is not as expected')

    def test_job_10_waiting_for_nodes(self):
        """ Testing the method waiting_for_nodes method that runs during federated training """

        responses = FakeResponses([
            {'node_id': 'node-1'}
        ])

        # Test False
        self.job._nodes = ['node-1']
        result = self.job.waiting_for_nodes(responses)
        self.assertFalse(result, 'wating_for_nodes method return True while expected is False')

        # Test True
        self.job._nodes = ['node-1', 'node-2']
        result = self.job.waiting_for_nodes(responses)
        self.assertTrue(result, 'waiting_for_nodes method return False while expected is True')

        responses = MagicMock(return_value=None)
        type(responses).dataframe = MagicMock(side_effect=KeyError)
        result = self.job.waiting_for_nodes(responses)
        self.assertTrue(result, 'waiting_for_nodes returned False while expected is False')

    @patch('fedbiomed.common.serializer.Serializer.load')
    @patch('fedbiomed.researcher.requests.Requests.send_message')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    @patch('fedbiomed.researcher.responses.Responses')
    def test_job_11_start_training_round(self,
                                         mock_responses,
                                         mock_requests_get_responses,
                                         mock_requests_send_message,
                                         serialize_load_patch,
                                         ):
        """ Test Job - start_training_round method with 3 different scenarios

            Test - 1 : When all the nodes successful completes training
            Test - 2 : When one of the nodes returns error during training
            Test - 3 : When one of the nodes returns error without extra_msg
                       This test also checks whether previous node (which returned)
                       error is removed or not
        """

        mock_responses.side_effect = TestJob.fake_responses_side_effect

        self.job._nodes = ['node-1', 'node-2']
        self.fds.data = MagicMock(return_value={
            'node-1': {'dataset_id': '1234'},
            'node-2': {'dataset_id': '12345'}
        })

        response_1 = {'node_id': 'node-1', 'researcher_id': environ['RESEARCHER_ID'],
                      'job_id': self.job._id, 'params_url': 'http://test.test',
                      'timing': {'rtime_total': 12},
                      'success': True,
                      'msg': 'MSG',
                      'dataset_id': '1234',
                      'command': 'train',
                      'sample_size': 100,
                      }

        response_2 = {'node_id': 'node-2', 'researcher_id': environ['RESEARCHER_ID'],
                      'job_id': self.job._id, 'params_url': 'http://test.test',
                      'timing': {'rtime_total': 12},
                      'success': True,
                      'msg': 'MSG',
                      'dataset_id': '1234',
                      'command': 'train',
                      'sample_size': 100,
                      }

        response_3 = {'node_id': 'node-2', 'researcher_id': environ['RESEARCHER_ID'],
                      'errnum': ErrorNumbers.FB100,
                      'extra_msg': 'this extra msg',
                      'command': 'error',
                      'dataset_id': '1234',
                      'sample_size': 100,
                      }

        response_4 = {'node_id': 'node-2', 'researcher_id': environ['RESEARCHER_ID'],
                      'extra_msg': False,
                      'errnum': ErrorNumbers.FB100,
                      'command': 'error',
                      'dataset_id': '1234',
                      'sample_size': 100,
                      }

        responses = FakeResponses([response_1, response_2])

        mock_requests_get_responses.return_value = responses
        aggregator_args = {node_id: {'aggregator_name': 'my_aggregator'} for node_id in self.job._nodes}
        # Test - 1
        nodes = self.job.start_nodes_training_round(1, aggregator_args_thr_msg=aggregator_args,
                                                    aggregator_args_thr_files={})
        _ = mock_requests_send_message.call_args_list
        self.assertEqual(mock_requests_send_message.call_count, 2)
        self.assertListEqual(nodes, ['node-1', 'node-2'])
        self.assertEqual(serialize_load_patch.call_count, 2)

        # Test - 2 When one of the nodes returns error
        mock_requests_send_message.reset_mock()
        serialize_load_patch.reset_mock()
        responses = FakeResponses([response_1, response_3])
        mock_requests_get_responses.return_value = responses
        nodes = self.job.start_nodes_training_round(2, aggregator_args_thr_msg=aggregator_args,
                                                    aggregator_args_thr_files={})
        self.assertEqual(mock_requests_send_message.call_count, 2)
        self.assertListEqual(nodes, ['node-1'])
        self.assertEqual(serialize_load_patch.call_count, 1)  # resp_3 has no params

        # Test - 2 When one of the nodes returns error without extra_msg and
        # check node-2 is removed since it returned error in the previous test call
        serialize_load_patch.reset_mock()
        mock_requests_send_message.reset_mock()
        responses = FakeResponses([response_1, response_4])
        mock_requests_get_responses.return_value = responses
        nodes = self.job.start_nodes_training_round(3, aggregator_args_thr_msg=aggregator_args,
                                                    aggregator_args_thr_files={})
        self.assertEqual(mock_requests_send_message.call_count, 1)
        self.assertListEqual(nodes, ['node-1'])
        self.assertEqual(serialize_load_patch.call_count, 1)

    def test_job_12_start_nodes_training_round_optim_aux_var(self):
        """Test that 'optim_aux_var' is properly used in 'start_nodes_training_round'."""
        fake_aux_var = {"module": {"key": "val"}}
        # General setup: skip requests sending and replies processing.
        self.job.nodes = []
        with (
            patch('fedbiomed.researcher.requests.Requests.send_message'),
            patch.object(self.job, "waiting_for_nodes", autospec=True)
                as patch_waiting_for_nodes,
        ):
            patch_waiting_for_nodes.return_value = False
            # Test that the aux-var upload is called when aux vars are passed.
            with patch.object(
                self.job, "upload_agg_optimizer_aux_var", autospec=True
            ) as patch_upload_agg_optimizer_aux_var:
                patch_upload_agg_optimizer_aux_var.return_value = (None, {})
                self.job.start_nodes_training_round(
                    1, {}, {}, do_training=True, optim_aux_var=fake_aux_var
                )
                patch_upload_agg_optimizer_aux_var.assert_called_once_with(
                    fake_aux_var
                )
            # Test that the aux-var upload is not called without input aux var.
            with patch.object(
                self.job, "upload_agg_optimizer_aux_var", autospec=True
            ) as patch_upload_agg_optimizer_aux_var:
                self.job.start_nodes_training_round(
                    1, {}, {}, do_training=True, optim_aux_var=None
                )
                patch_upload_agg_optimizer_aux_var.assert_not_called()
            # Test that the aux-var upload is not called in evaluation mode.
            with patch.object(
                self.job, "upload_agg_optimizer_aux_var", autospec=True
            ) as patch_upload_agg_optimizer_aux_var:
                self.job.start_nodes_training_round(
                    1, {}, {}, do_training=False, optim_aux_var=fake_aux_var
                )
                patch_upload_agg_optimizer_aux_var.assert_not_called()

    def test_job_13_update_parameters_with_invalid_arguments(self):
        """Testing update_parameters method with invalid arguments."""
        # Reset calls that comes from init time
        self.mock_upload_file.reset_mock()
        params = {'params': [1, 2, 3, 4]}
        # Test by passing both params and filename: raises.
        with self.assertRaises(SystemExit):
            self.job.update_parameters(params=params, filename='dummy/file/name/')
        # Test without passing parameters should raise ValueError
        with self.assertRaises(SystemExit):
            self.job.update_parameters()

    def test_job_14_update_parameters_from_params(self):
        """Testing update_parameters when passing 'params'."""
        params = {'params': [1, 2, 3, 4]}
        with (
            patch("fedbiomed.common.serializer.Serializer.dump") as patch_dump,
            patch.object(self.job.training_plan, "get_model_params") as patch_get
        ):
            patch_get.return_value = params
            result = self.job.update_parameters(params=params)
        self.assertEqual((self.job._model_params_file, self.job.repo.uploads_url) , result)
        self.model.get_model_params.assert_called_once()
        patch_dump.assert_called_with(
            {"researcher_id": self.job._researcher_id, "model_weights": params},
            self.job._model_params_file,
        )

    def test_job_15_update_parameters_from_file(self):
        """Testing update_parameters when passing 'filename'."""
        params = {"params": [1, 2, 3, 4]}
        with (
            patch("fedbiomed.common.serializer.Serializer.load") as patch_load
        ):
            patch_load.return_value = {"researcher_id": 1234, "model_weights": params}
            result = self.job.update_parameters(filename="mock_path")
        patch_load.assert_called_with("mock_path")
        self.model.set_model_params.assert_called_once_with(params)
        self.assertEqual((self.job._model_params_file, self.job.repo.uploads_url) , result)

    @patch('uuid.uuid4', autospec=True)
    @patch('fedbiomed.common.serializer.Serializer.dump', autospec=True)
    def test_job_16_upload_agg_optimizer_aux_var(
        self,
        patch_serializer_dump,
        patch_uuid,
    ):
        """Test 'upload_agg_optimizer_aux_var' with both shared and node-specific info."""
        # Set up: two target nodes; aux vars with shared and node-specific info.
        self.job.nodes = ["node-1", "node-2"]
        setattr(self.job, "_keep_files_dir", "dir")  # simplify dump paths
        aux_var = {
            "module_a": {"key": "val"},
            "module_b": {f"node-{i + 1}": {"key": "val"} for i in range(3)},
        }
        patch_uuid.return_value = "uuid"
        self.mock_upload_file.reset_mock()  # resetting fedbiomed.common.repository.Repository.upload_file patcher
        self.mock_upload_file.return_value = {"file": 'url'}

        # Call the tested method.
        url_shared, url_bynode = self.job.upload_agg_optimizer_aux_var(aux_var)
        # Verify that results and mock calls match expectations.
        self.assertEqual(url_shared, "url")
        self.assertDictEqual(url_bynode, {"node-1": "url", "node-2": "url"})
        patch_serializer_dump.assert_has_calls([
            call({"module_a": {"key": "val"}}, "dir/aux_var_shared_uuid.mpk"),
            call({"module_b": {"key": "val"}}, "dir/aux_var_node_node-1_uuid.mpk"),
            call({"module_b": {"key": "val"}}, "dir/aux_var_node_node-2_uuid.mpk"),
        ], any_order=True)
        self.mock_upload_file.assert_has_calls([
            call("dir/aux_var_shared_uuid.mpk"),
            call("dir/aux_var_node_node-1_uuid.mpk"),
            call("dir/aux_var_node_node-2_uuid.mpk"),
        ], any_order=True)

    @patch('uuid.uuid4', autospec=True)
    @patch('fedbiomed.common.serializer.Serializer.dump', autospec=True)
    def test_job_17_upload_agg_optimizer_aux_var_shared_only(
        self,
        patch_serializer_dump,
        patch_uuid,
    ):
        """Test 'upload_agg_optimizer_aux_var' with shared info only."""
        # Set up: two target nodes; aux vars with shared and node-specific info.
        self.job.nodes = ["node-1", "node-2"]
        setattr(self.job, "_keep_files_dir", "dir")  # simplify dump paths
        aux_var = {
            "module_a": {"key": "val"},
            "module_b": {"key": "val"},
        }
        fake_url = "url"

        patch_uuid.return_value = "uuid"

        self.mock_upload_file.reset_mock()  # resetting fedbiomed.common.repository.Repository.upload_file patcher
        self.mock_upload_file.return_value = {"file": fake_url}

        expected_file_path = "dir/aux_var_shared_uuid.mpk"
        # Call the tested method.
        url_shared, url_bynode = self.job.upload_agg_optimizer_aux_var(aux_var)
        # Verify that results and mock calls match expectations.
        self.assertEqual(url_shared, fake_url)
        self.assertDictEqual(url_bynode, {})
        patch_serializer_dump.assert_called_once_with(
            aux_var, expected_file_path
        )
        self.mock_upload_file.assert_called_once_with(expected_file_path)

    @patch('uuid.uuid4', autospec=True)
    @patch('fedbiomed.common.serializer.Serializer.dump', autospec=True)
    def test_job_18_upload_agg_optimizer_aux_var_bynode_only(
        self,
        patch_serializer_dump,
        patch_uuid,
    ):
        """Test 'upload_agg_optimizer_aux_var' with node-specific info only."""
        # Set up: two target nodes; aux vars with shared and node-specific info.
        self.job.nodes = ["node-1", "node-2"]
        setattr(self.job, "_keep_files_dir", "dir")  # simplify dump paths
        aux_var = {
            "module_a": {f"node-{i + 1}": {"key": "val"} for i in range(2)},
            "module_b": {f"node-{i + 1}": {"key": "val"} for i in range(2)},
        }
        patch_uuid.return_value = "uuid"
        self.mock_upload_file.reset_mock()
        self.mock_upload_file.return_value = {"file": "url"}
        # Call the tested method.
        url_shared, url_bynode = self.job.upload_agg_optimizer_aux_var(aux_var)
        # Verify that results and mock calls match expectations.
        self.assertIsNone(url_shared)
        self.assertDictEqual(url_bynode, {"node-1": "url", "node-2": "url"})
        aux_var_node = {"module_a": {"key": "val"}, "module_b": {"key": "val"}}
        patch_serializer_dump.assert_has_calls([
            call(aux_var_node, "dir/aux_var_node_node-1_uuid.mpk"),
            call(aux_var_node, "dir/aux_var_node_node-2_uuid.mpk"),
        ], any_order=True)
        self.mock_upload_file.assert_has_calls([
            call("dir/aux_var_node_node-1_uuid.mpk"),
            call("dir/aux_var_node_node-2_uuid.mpk"),
        ], any_order=True)

    @patch('fedbiomed.common.logger.logger.error')
    def test_job_19_check_dataset_quality(self, mock_logger_error):
        """ Test for checking data quality in Job by providing different FederatedDatasets """

        # CSV - Check dataset when everything is okay
        self.fds.data.return_value = {
            'node-1': {'data_type': 'csv', 'dtypes': ['float', 'float', 'float'], 'shape': [10, 5]},
            'node-2': {'data_type': 'csv', 'dtypes': ['float', 'float', 'float'], 'shape': [10, 5]}
        }
        try:
            self.job.check_data_quality()
        except:
            self.assertTrue(True, 'Raised error when given CSV datasets are OK')

        # CSV - Check when data types are different
        self.fds.data.return_value = {
            'node-1': {'data_type': 'csv', 'dtypes': ['float', 'float', 'float'], 'shape': [10, 5]},
            'node-2': {'data_type': 'image', 'dtypes': ['float', 'float', 'float'], 'shape': [10, 5]}
        }
        with self.assertRaises(Exception):
            self.job.check_data_quality()

        # CSV - Check when dimensions are different
        self.fds.data.return_value = {
            'node-1': {'data_type': 'csv', 'dtypes': ['float', 'float', 'float'], 'shape': [10, 15]},
            'node-2': {'data_type': 'csv', 'dtypes': ['float', 'float', 'float'], 'shape': [10, 5]}
        }
        with self.assertRaises(Exception):
            self.job.check_data_quality()

        # CSV - Check when dtypes do not match
        self.fds.data.return_value = {
            'node-1': {'data_type': 'csv', 'dtypes': ['float', 'int', 'float'], 'shape': [10, 15]},
            'node-2': {'data_type': 'csv', 'dtypes': ['int', 'float', 'float'], 'shape': [10, 5]}
        }
        with self.assertRaises(Exception):
            self.job.check_data_quality()

        # Image Dataset - Check when datasets are OK
        self.fds.data.return_value = {
            'client-1': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 3, 10, 10]},
            'client-2': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 3, 10, 10]},
        }
        try:
            self.job.check_data_quality()
        except:
            self.assertTrue(True, 'Raised error when given datasets are OK')

        # Image Dataset - Check when color channels do not match
        self.fds.data.return_value = {
            'client-1': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 3, 10, 10]},
            'client-2': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 5, 10, 10]},
        }
        # Logs error instead of raising error
        mock_logger_error.reset_mock()
        self.job.check_data_quality()
        mock_logger_error.assert_called_once()

        # Image Dataset - Check when dimensions do not match
        self.fds.data.return_value = {
            'client-1': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 3, 16, 10]},
            'client-2': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 3, 10, 10]},
        }
        # Logs error instead of raising error
        mock_logger_error.reset_mock()
        self.job.check_data_quality()
        mock_logger_error.assert_called_once()

        # Image Dataset - Check when dimensions and color channels do not match
        self.fds.data.return_value = {
            'client-1': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 3, 16, 10]},
            'client-2': {'data_type': 'images', 'dtypes': [], 'shape': [1000, 5, 10, 10]},
        }
        # Logs error instead of raising error
        mock_logger_error.reset_mock()
        self.job.check_data_quality()
        self.assertEqual(mock_logger_error.call_count, 2)

    def test_job_20_save_private_training_replies(self):
        """
        tests if `_save_training_replies` is properly extracting
        breakpoint info from `training_replies`. It uses a dummy class
        FakeResponses, a weak implementation of `Responses` class
        """

        # first create a `_training_replies` variable
        training_replies = {
            0: FakeResponses([]),
            1: FakeResponses(
                [
                    {
                        "node_id": '1234',
                        'params': torch.Tensor([1, 3, 5]),
                        'dataset_id': 'id_node_1'
                    },
                    {
                        "node_id": '5678',
                        'params': np.array([1, 3, 5]),
                        'dataset_id': 'id_node_2'
                    },
                ])
        }

        # action
        new_training_replies = self.job._save_training_replies(training_replies)

        # check if `training_replies` is  saved accordingly
        self.assertTrue(type(new_training_replies) is list)
        self.assertTrue(len(new_training_replies) == 2)
        self.assertTrue('params' not in new_training_replies[1][0])
        self.assertEqual(new_training_replies[1][1].get('dataset_id'), 'id_node_2')

    @patch('fedbiomed.researcher.responses.Responses.__getitem__')
    @patch('fedbiomed.researcher.responses.Responses.__init__')
    def test_job_21_private_load_training_replies(
            self,
            patch_responses_init,
            patch_responses_getitem
    ):
        """tests if `_load_training_replies` is loading file content from path file
        and is building a proper training replies structure from breakpoint info
        """
        # Declare mock model parameters, for torch and scikit-learn.
        pytorch_params = {
            # dont need other fields
            "model_weights": torch.Tensor([1, 3, 5, 7])
        }
        sklearn_params = {
            # dont need other fields
            "model_weights": np.array([[1, 2, 3, 4, 5], [2, 8, 7, 5, 5]])
        }
        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value={})

        # mock Responses
        #
        # nota: works fine only with one instance of Response active at a time thus
        # - cannot be used in `test_save_private_training_replies`
        # - if testing on more than 1 round, only the last round can be used for Asserts
        def side_responses_init(data, *args):
            self.responses_data = data

        def side_responses_getitem(arg, *args):
            return self.responses_data[arg]

        patch_responses_init.side_effect = side_responses_init
        patch_responses_init.return_value = None
        patch_responses_getitem.side_effect = side_responses_getitem

        # instantiate job with a mock training plan
        test_job_torch = Job(
            training_plan_class=MagicMock(),
            training_args=training_args_for_testing,
            data=fds
        )
        # second create a `training_replies` variable
        loaded_training_replies_torch = [
            [
                {"success": True,
                 "msg": "",
                 "dataset_id": "dataset_1234",
                 "node_id": "node_1234",
                 "params_path": "/path/to/file/param.mpk",
                 "timing": {"time": 0}
                 },
                {"success": True,
                 "msg": "",
                 "dataset_id": "dataset_4567",
                 "node_id": "node_4567",
                 "params_path": "/path/to/file/param2.mpk",
                 "timing": {"time": 0}
                 }
            ]
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
            loaded_training_replies_torch[0][1]["params_path"],
        )
        self.assertIsInstance(torch_training_replies, dict)
        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            torch_training_replies[0][0]['node_id'],
            loaded_training_replies_torch[0][0]['node_id'])
        # check `training_replies` for pytorch models
        self.assertTrue(torch.eq(
            torch_training_replies[0][1]['params'],
            pytorch_params['model_weights']
        ).all())
        self.assertEqual(
            torch_training_replies[0][1]['params_path'],
            "/path/to/file/param2.mpk"
        )
        self.assertTrue(isinstance(torch_training_replies[0], Responses))

        # #### REPRODUCE TESTS BUT FOR SKLEARN MODELS AND 2 ROUNDS
        # create a `training_replies` variable
        loaded_training_replies_sklearn = [
            [
                {
                    # dummy
                    "params_path": "/path/to/file/param_sklearn.mpk"
                }
            ],
            [
                {"success": False,
                 "msg": "",
                 "dataset_id": "dataset_8888",
                 "node_id": "node_8888",
                 "params_path": "/path/to/file/param2_sklearn.mpk",
                 "timing": {"time": 6}
                 }
            ]
        ]

        # instantiate job
        test_job_sklearn = Job(
            training_plan_class=MagicMock(),
            training_args=training_args_for_testing,
            data=fds
        )

        # action
        with patch(
            "fedbiomed.common.serializer.Serializer.load", return_value=sklearn_params
        ) as load_patch:
            sklearn_training_replies = test_job_sklearn._load_training_replies(
                loaded_training_replies_sklearn
            )
        self.assertEqual(load_patch.call_count, 2)
        load_patch.assert_called_with(
            loaded_training_replies_sklearn[1][0]["params_path"],
        )
        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            sklearn_training_replies[1][0]['node_id'],
            loaded_training_replies_sklearn[1][0]['node_id'])
        # check `training_replies` for sklearn models
        self.assertTrue(np.allclose(
            sklearn_training_replies[1][0]['params'],
            sklearn_params['model_weights']
        ))
        self.assertEqual(
            sklearn_training_replies[1][0]['params_path'],
            "/path/to/file/param2_sklearn.mpk"
        )
        self.assertTrue(isinstance(sklearn_training_replies[0],
                                   Responses))

    @patch('fedbiomed.researcher.job.Job._load_training_replies')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    def test_job_22_load_state(
            self,
            patch_job_update_parameters,
            patch_job_load_training_replies
    ):
        """
        test if the job state values correctly initialize job
        """

        job_state = {
            'researcher_id': 'my_researcher_id_123456789',
            'job_id': 'my_job_id_abcdefghij',
            'model_params_path': '/path/to/my/model_file.py',
            'training_replies': {0: 'un', 1: 'deux'}
        }
        new_training_replies = {2: 'trois', 3: 'quatre'}

        # patch `update_parameters`
        patch_job_update_parameters.return_value = "dummy_string"

        # patch `_load_training_replies`
        patch_job_load_training_replies.return_value = new_training_replies

        # action
        self.job.load_state(job_state)

        self.assertEqual(self.job._researcher_id, job_state['researcher_id'])
        self.assertEqual(self.job._id, job_state['job_id'])
        self.assertEqual(self.job._training_replies, new_training_replies)

    @patch('fedbiomed.researcher.job.create_unique_link')
    @patch('fedbiomed.researcher.job.create_unique_file_link')
    @patch('fedbiomed.researcher.job.Job._save_training_replies')
    def test_job_23_save_state(
            self,
            patch_job_save_training_replies,
            patch_create_unique_file_link,
            patch_create_unique_link
    ):
        """
        test that job breakpoint state structure + file links are created
        """

        new_training_replies = [
            [
                {'params_path': '/path/to/job_test_save_state_params0.pt'}
            ],
            [
                {'params_path': '/path/to/job_test_save_state_params1.pt'}
            ]
        ]
        # expected transformed values of new_training_replies for save state
        new_training_replies_state = [
            [
                {'params_path': 'xxx/job_test_save_state_params0.pt'}
            ],
            [
                {'params_path': 'xxx/job_test_save_state_params1.pt'}
            ]
        ]

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
        save_state = self.job.save_state(breakpoint_path)

        self.assertEqual(environ['RESEARCHER_ID'], save_state['researcher_id'])
        self.assertEqual(self.job._id, save_state['job_id'])
        self.assertEqual(link_path, save_state['model_params_path'])
        # check transformation of training replies
        for round_i, round in enumerate(new_training_replies):
            for response_i, _ in enumerate(round):
                self.assertEqual(
                    save_state['training_replies'][round_i][response_i]['params_path'],
                    new_training_replies_state[round_i][response_i]['params_path'])

    def test_job_24_upload_aggregator_args(self):
        training_args_thr_msg = {'node-1': {'var1': 1, 'var2': [1, 2]},
                                 'node-2': {'var1': 1, 'var2': [1, 2]}}
        tensor = torch.Tensor([[1, 2, 4], [2, 3, 4]])
        arr = np.array([1, 4, 5])
        training_args_thr_files = {'node-1':{ 'aggregator_name': 'my_aggregator',
                                    'var4': {'params': tensor}, 'var5': {'params': arr}},
                                   'node-2':{ 'aggregator_name': 'my_aggregator',
                                             'var4': {'params': tensor.T}, 'var5': {'params':arr}}
                                   }
        with patch.object(uuid, 'uuid4') as patch_uuid:
            patch_uuid.return_value = FakeUuid()
            t_a = self.job.upload_aggregator_args(copy.deepcopy(training_args_thr_msg), training_args_thr_files)
            # first we check `training_args_thr_msg` are contained into `t_a` (be careful about references!)
            self.assertEqual(training_args_thr_msg, t_a | training_args_thr_msg  )
            print(t_a)
            # then, check parameters are updated into `training_args_thr_msg`
            for node_id in ('node-1', 'node-2'):
                for var in ('var4', 'var5'):
                    # check that `t_a` doesnot contain any params field
                    self.assertIsNone(t_a[node_id][var].get('params'))
                    filename = os.path.join(self.job._keep_files_dir, f"{var}_{FakeUuid.VALUE}.mpk")
                    self.assertEqual(t_a[node_id][var]['filename'], filename)
                    self.assertEqual(t_a[node_id][var]['url'], self.job.repo.uploads_url)

    def test_job_25_extract_received_optimizer_aux_var_from_round(self):
        """Test that 'extract_received_optimizer_aux_var_from_round' works well."""
        # Set up: nodes sent back some Optimizer aux var information.
        responses = Responses([])
        responses.append(Responses({
            "node_id": "node-1",
            "optim_aux_var": {
                "module_a": {"key": "a1"}, "module_b": {"key": "b1"}
            },
        }))
        responses.append(Responses({
            "node_id": "node-2",
            "optim_aux_var": {
                "module_a": {"key": "a2"}, "module_b": {"key": "b2"}
            },
        }))
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
        responses = Responses([])
        responses.append(Responses({"node_id": "node-1"}))
        responses.append(Responses({"node_id": "node-2"}))
        getattr(self.job, "_training_replies")[1] = responses
        # Call the method and verify that it returns an empty dict.
        aux_var = self.job.extract_received_optimizer_aux_var_from_round(round_id=1)
        self.assertDictEqual(aux_var, {})


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
