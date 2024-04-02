import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, create_autospec, patch

from testsupport.base_case import ResearcherTestCase  # Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_mocks import MockRequestModule
from testsupport.fake_training_plan import FakeTorchTrainingPlan
from testsupport import fake_training_plan

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.message import \
    TrainReply,ErrorMessage, TrainingPlanStatusReply
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import DiscardOnTimeout
from fedbiomed.researcher.federated_workflows.jobs import \
    Job, TrainingJob, TrainingPlanApproveJob, TrainingPlanCheckJob


class TestJob(ResearcherTestCase, MockRequestModule):
    """Tests Job class and all of its subclasses"""
    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows.jobs._job.Requests")
        self.patch_serializer = patch("fedbiomed.common.serializer.Serializer")
        self.mock_serializer = self.patch_serializer.start()

        # Globally create mock for Model and FederatedDataset
        self.model = create_autospec(BaseTrainingPlan, instance=False)
        self.fds = MagicMock(spec=FederatedDataSet)
        self.fds.data = MagicMock(return_value={})
        self.model = FakeTorchTrainingPlan
        self.model.save_code = MagicMock()

    def tearDown(self) -> None:

        self.patch_serializer.stop()

        # Remove if there is dummy model file
        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

        super().tearDown()

    def test_job_01_base_job(self):
        class MinimalJob(Job):
            def execute():
                pass

        nodes = MagicMock(spec=list)
        files_dir = '/path/to/my/files'
        job = MinimalJob(nodes=nodes, keep_files_dir=files_dir)
        self.assertIsNotNone(job._keep_files_dir)  # must be initialized by Job
        self.assertTrue(isinstance(job._nodes, list) and len(job._nodes) == 0)  # nodes must be empty list by default

        # Job can take nodes and keep_files_dir as arguments
        mynodes = ['first-node', 'second-node']
        job = MinimalJob(
            nodes = mynodes,
            keep_files_dir='keep_files_dir'
        )
        self.assertEqual(job._keep_files_dir, 'keep_files_dir')
        self.assertTrue(all(x == y for x, y in zip(job._nodes, mynodes)))

        # use and check timer
        with job.RequestTimer(mynodes) as t1:
            pass

        for node in mynodes:
            self.assertTrue(node in t1)
            self.assertTrue(isinstance(t1[node], float))
            self.assertAlmostEqual(t1[node], 0, delta=10**-3)

        # use and check getters
        n = job.nodes
        self.assertEqual(n, mynodes)

        r = job.requests
        # need to access the private member
        self.assertEqual(r, job._reqs)


    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.uuid.uuid4', return_value='UUID')
    def test_job_02_training_job_successful(self, mock_uuid):

        # Initializing a training plan instance via Job must call:
        # 1) the training plan's default constructor
        # 2) training plan's post init
        mock_tp_class = MagicMock()
        mock_tp_class.return_value = MagicMock(spec=BaseTrainingPlan)
        mock_tp_class.__name__ = 'mock_tp_class'

        mock_tp = mock_tp_class()
        mock_tp.get_model_params.return_value = MagicMock(spec=dict)
        mock_tp.source.return_value = MagicMock(spec=str)

        fake_node_state_ids = {
            'alice': 'alide_nsid',
            'bob': 'bob_nsid'
        }

        optim_aux_vars = [
            {
                'shared': {},
                'node-specific': {
                    'alice': 'node-specific',
                    'bob': 'node-specific'
                }
            },
            {}

        ]
        for optim_aux_var in optim_aux_vars:
            for do_training in [True, False]:
                # initialize TrainingJob
                with tempfile.TemporaryDirectory() as fp:
                    job = TrainingJob(
                        experiment_id='some_id',
                        round_=1,
                        training_plan=mock_tp,
                        training_args=TrainingArgs({}, only_required=False),
                        model_args=None,
                        data=self.fds,  # mocked FederatedDataSet class
                        nodes_state_ids=fake_node_state_ids,
                        nodes = ['alice', 'bob'],
                        aggregator_args={},
                        do_training=do_training,
                        optim_aux_var=optim_aux_var,
                        keep_files_dir=fp
                    )

                    # Calling execute() must:
                    # 1) call the `Requests.send` function to initiate training on the nodes
                    # 2) return the properly formatted replies
                    #
                    # job._nodes = ['alice', 'bob']
                    self.fds.data = MagicMock(return_value={
                        'alice': {'dataset_id': 'alice_data'},
                        'bob': {'dataset_id': 'bob_data'},
                    })

                    self.mock_federated_request.errors.return_value = {}
                    self.mock_federated_request.replies.return_value = {
                        'alice': TrainReply(**self._get_train_reply(
                            'alice',
                            self.fds.data()['alice']['dataset_id'],
                            {'module': 'params_alice'})),
                        'bob': TrainReply(**self._get_train_reply(
                            'bob',
                            self.fds.data()['bob']['dataset_id'],
                            {'module': 'params_bob'})),
                    }
                    with patch("time.perf_counter") as mock_perf_counter:
                        mock_perf_counter.return_value = 0
                        training_replies, aux_vars = job.execute()

                    # Following line tests if aux_vars from training replies extracted correctly
                    # aux_vars are set only if traning is done
                    if do_training:
                        self.assertDictEqual(aux_vars, {'module': {'alice': 'params_alice', 'bob': 'params_bob'}})
                    else:
                        self.assertEqual(aux_vars, None)

                    self.mock_requests.return_value.send.called_once_with(
                        [
                            (
                                {'alice': self._get_train_request(
                                    mock_tp, {}, 'alice', fake_node_state_ids, self.fds.data()),
                                 'bob': self._get_train_request(mock_tp, {}, 'bob', fake_node_state_ids, self.fds.data())},
                                ['alice', 'bob']
                            )
                        ]
                    )

                    # populate expected replies
                    expected_replies = {}
                    for node_id, r in self.mock_federated_request.replies.return_value.items():
                        expected_replies.update({
                            node_id: {
                                **r.get_dict(),
                                'params_path': os.path.join(job._keep_files_dir, f"params_{node_id}_{mock_uuid.return_value}.mpk")
                            }
                        })
                    self.assertDictEqual(training_replies, expected_replies)


    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.uuid.uuid4', return_value='UUID')
    def test_job_03_training_job_failed(self, mock_uuid):

        # Initializing a training plan instance via Job must call:
        # 1) the training plan's default constructor
        # 2) training plan's post init
        mock_tp_class = MagicMock()
        mock_tp_class.return_value = MagicMock(spec=BaseTrainingPlan)
        mock_tp_class.__name__ = 'mock_tp_class'

        mock_tp = mock_tp_class()
        mock_tp.get_model_params.return_value = MagicMock(spec=dict)
        mock_tp.source.return_value = MagicMock(spec=str)

        fake_node_state_ids = {
            'alice': 'alide_nsid',
            'bob': 'bob_nsid'
        }
        error_status_all = [
            {
                'alice': True,
                'bob': True,
            },
            {
                'alice': False,
                'bob': True,
            },
            {
                'alice': True,
                'bob': False,
            },
            {
                'alice': False,
                'bob': False,
            },
        ]
        success_status_all = [
            {
                'alice': True,
                'bob': False,
            },
            {
                'alice': False,
                'bob': True,
            },
            {
                'alice': False,
                'bob': False,
            },
        ]
        aux_var_return = {
            'alice': {'module': 'params_alice'},
            'bob': {'module': 'params_bob'},
        }

        # Test all cases with error or with an unsuccessful training reply
        for error_status in error_status_all:
            for success_status in success_status_all:
                # initialize TrainingJob
                with tempfile.TemporaryDirectory() as fp:
                    job = TrainingJob(
                        experiment_id='some_id',
                        round_=1,
                        training_plan=mock_tp,
                        training_args=TrainingArgs({}, only_required=False),
                        model_args=None,
                        data=self.fds,  # mocked FederatedDataSet class
                        nodes_state_ids=fake_node_state_ids,
                        nodes = ['alice', 'bob'],
                        aggregator_args={},
                        do_training=True,
                        optim_aux_var={},
                        keep_files_dir=fp
                    )

                    # Calling execute() must:
                    # 1) call the `Requests.send` function to initiate training on the nodes
                    # 2) return the properly formatted replies
                    #
                    # job._nodes = ['alice', 'bob']
                    self.fds.data = MagicMock(return_value={
                        'alice': {'dataset_id': 'alice_data'},
                        'bob': {'dataset_id': 'bob_data'},
                    })

                    # REplies / errors depend on tested scenario
                    err = {}
                    ret = {}
                    for node in ['alice', 'bob']:
                        if not error_status[node]:
                            ret[node] = TrainReply(**self._get_train_reply(
                                node,
                                self.fds.data()[node]['dataset_id'],
                                aux_var_return[node],
                                success=success_status[node]))
                        else:
                            err[node] = ErrorMessage(**self._get_error_message(node))
                    self.mock_federated_request.replies.return_value = ret
                    self.mock_federated_request.errors.return_value = err

                    # Execute the payload
                    with patch("time.perf_counter") as mock_perf_counter:
                        mock_perf_counter.return_value = 0
                        training_replies, aux_vars = job.execute()

                    # Following line tests if aux_vars from training replies extracted correctly
                    # aux_vars are set only if traning is done
                    expected_aux_vars = {}
                    for node in ['alice', 'bob']:
                        if not error_status[node] and success_status[node]:
                            expected_aux_vars.setdefault('module', {})[node] = aux_var_return[node]['module']
                    self.assertDictEqual(aux_vars, expected_aux_vars)

                    self.mock_requests.return_value.send.called_once_with(
                        [
                            (
                                {'alice': self._get_train_request(
                                    mock_tp, {}, 'alice', fake_node_state_ids, self.fds.data()),
                                 'bob': self._get_train_request(mock_tp, {}, 'bob', fake_node_state_ids, self.fds.data())},
                                ['alice', 'bob']
                            )
                        ]
                    )

                    # populate expected replies
                    expected_replies = {}
                    for node_id, r in self.mock_federated_request.replies.return_value.items():
                        if not error_status[node_id] and success_status[node_id]:
                            expected_replies.update({
                                node_id: {
                                    **r.get_dict(),
                                    'params_path': os.path.join(job._keep_files_dir, f"params_{node_id}_{mock_uuid.return_value}.mpk")
                                }
                            })
                    self.assertDictEqual(training_replies, expected_replies)

                    # minimal: check errors were recovered (no very significant test)
                    self.mock_federated_request.errors.assert_called_once()
                    self.mock_federated_request.reset_mock()


    @patch('fedbiomed.researcher.federated_workflows.jobs._training_plan_approval_job.DiscardOnTimeout')
    def test_job_04_training_plan_approve_job(self, mock_policy_dot):

        mock_policy_dot = MagicMock(spec=DiscardOnTimeout)

        mock_tp_class = MagicMock()
        mock_tp_class.return_value = MagicMock(spec=BaseTrainingPlan)
        mock_tp_class.__name__ = 'mock_tp_class'

        mock_tp = mock_tp_class()
        mock_tp.get_model_params.return_value = MagicMock(spec=dict)
        mock_tp.source.return_value = MagicMock(spec=str)

        success_status_all =[
            {'alice': True, 'bob': True},
            {'alice': True, 'bob': False},
            {'alice': False, 'bob': True},
            {'alice': False, 'bob': False},
        ]


        for success_status in success_status_all:
            # initialize TrainingJob
            with tempfile.TemporaryDirectory() as fp:
                job = TrainingPlanApproveJob(
                    training_plan=mock_tp,
                    description='my test TP',
                    nodes = ['alice', 'bob'],
                    keep_files_dir=fp
                )

                # prepare mocked node answers
                self.mock_requests.return_value.training_plan_approve.return_value = success_status

                # execute the tested payload
                with patch("time.perf_counter") as mock_perf_counter:
                    mock_perf_counter.return_value = 0
                    approval_replies = job.execute()

                # check call of message sending to nodes
                self.mock_requests.return_value.training_plan_approve.called_once_with(
                    [
                        (
                            mock_tp,
                            'any arbitrary message',
                            ['alice', 'bob'],
                            None
                        )
                    ]
                )

                # check received the expected answers
                self.assertDictEqual(approval_replies, self.mock_requests.return_value.training_plan_approve.return_value)


    @patch('fedbiomed.researcher.federated_workflows.jobs._training_plan_approval_job.DiscardOnTimeout')
    def test_job_05_training_plan_check_job(self,
                                            mock_policy_dot):

        mock_policy_dot = MagicMock(spec=DiscardOnTimeout)

        mock_tp_class = MagicMock()
        mock_tp_class.return_value = MagicMock(spec=BaseTrainingPlan)
        mock_tp_class.__name__ = 'mock_tp_class'

        mock_tp = mock_tp_class()
        mock_tp.get_model_params.return_value = MagicMock(spec=dict)
        mock_tp.source.return_value = MagicMock(spec=str)

        error_status_all = [
            {'alice': True, 'bob': True},
            {'alice': True, 'bob': False},
            {'alice': False, 'bob': True},
            {'alice': False, 'bob': False},
        ]

        success_status_all = [
            {'alice': True, 'bob': True},
            {'alice': True, 'bob': False},
            {'alice': False, 'bob': True},
            {'alice': False, 'bob': False},
        ]

        # test for a matrx of cases / scenarios
        for error_status in error_status_all:
            for approval_obligation in [True, False]:
                for success_status in success_status_all:
                    for alice_approval_status in TrainingPlanApprovalStatus:
                        for bob_approval_status in TrainingPlanApprovalStatus:
                            # initialize TrainingJob
                            with tempfile.TemporaryDirectory() as fp:
                                job = TrainingPlanCheckJob(
                                    experiment_id='any_unused_id',
                                    training_plan=mock_tp,
                                    nodes = ['alice', 'bob'],
                                    keep_files_dir=fp
                                )

                                err = {}
                                ret = {}
                                approval_status = {
                                    'alice': alice_approval_status,
                                    'bob': bob_approval_status,
                                }

                                self.mock_federated_request.errors.return_value = {}
                                self.mock_federated_request.replies.return_value = {
                                    'alice': TrainingPlanStatusReply(**self._get_status_reply(
                                        mock_tp,
                                        'alice',
                                        success_status['alice'],
                                        approval_obligation,
                                        alice_approval_status.value)),
                                    'bob': TrainingPlanStatusReply(**self._get_status_reply(
                                        mock_tp,
                                        'bob',
                                        success_status['bob'],
                                        approval_obligation,
                                        bob_approval_status.value)),
                                }
                                for node in ['alice', 'bob']:
                                    if error_status[node]:
                                        err[node] = ErrorMessage(**self._get_error_message(node))
                                    else:
                                        ret[node] = TrainingPlanStatusReply(**self._get_status_reply(
                                            mock_tp,
                                            node,
                                            success_status[node],
                                            approval_obligation,
                                            approval_status[node].value)
                                        )

                                self.mock_federated_request.errors.return_value = err
                                self.mock_federated_request.replies.return_value = ret

                                # execute tested payload
                                with patch("time.perf_counter") as mock_perf_counter:
                                    mock_perf_counter.return_value = 0
                                    check_replies = job.execute()

                                # checks
                                self.mock_requests.return_value.send.called_once_with(
                                    [
                                        (
                                            {'alice': self._get_status_request(mock_tp),
                                             'bob': self._get_status_request(mock_tp)},
                                            ['alice', 'bob']
                                        )
                                    ]
                                )

                                self.assertDictEqual(check_replies, self.mock_federated_request.replies.return_value)

                                # we lack a test using the errors but no obvious test for this case


    def _get_train_request(self,
                           mock_tp,
                           secagg_arguments,
                           node_id,
                           state_ids,
                           data):
        return {
            'request_id': 'this_request',
            'researcher_id': environ['RESEARCHER_ID'],
            'experiment_id': 'some_id',
            'training_args': {},
            'training': True,
            'model_args': {},
            'round': 1,
            'training_plan': mock_tp.source(),
            'training_plan_class': mock_tp.__class__.__name__,
            'params': mock_tp.get_model_params(),
            'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
            'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
            'secagg_random': secagg_arguments.get('secagg_random'),
            'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
            'command': 'train',
            'aggregator_args': {},
            'aux_vars': [{}, 'node-specific'],
            'state_id': state_ids[node_id],
            'dataset_id': data[node_id]['dataset_id'],
        }

    def _get_train_reply(self,
                         node_id,
                         dataset_id,
                         optim_aux_var,
                         success=True):
        return {
            'request_id': 'this_request',
            'researcher_id': environ['RESEARCHER_ID'],
            'experiment_id': 'some_id',
            'success': success,
            'node_id': node_id,
            'dataset_id': dataset_id,
            'timing': {'rtime_total': 0},
            'msg': '',
            'command': 'train',
            'state_id': None,
            'sample_size': None,
            'encrypted': False,
            'params': None,
            'optimizer_args': None,
            'optim_aux_var': optim_aux_var,
            'encryption_factor': None,
        }

    def _get_error_message(self,
                           node_id):
        return {
            'request_id': 'this_request',
            'researcher_id': environ['RESEARCHER_ID'],
            'node_id': node_id,
            'errnum': 'a dummy error',
            'extra_msg': 'a dummy message',
            'command': 'error',
        }

    def _get_status_request(self,
                            mock_tp,
                            ):
        return {
            'request_id': 'this_request',
            'researcher_id': environ['RESEARCHER_ID'],
            'experiment_id': 'some_id',
            'training_plan': mock_tp.source(),
            'command': 'training-plan-status',
        }

    def _get_status_reply(self,
                          mock_tp,
                          node_id,
                          success=True,
                          approval_obligation=True,
                          status='the TP approval status'):
        return {
            'request_id': 'this_request',
            'researcher_id': environ['RESEARCHER_ID'],
            'node_id': node_id,
            'experiment_id': 'some_id',
            'success': success,
            'approval_obligation': approval_obligation,
            'status': status,
            'msg': 'my arbitrary message',
            'training_plan': mock_tp.source(),
            'command': 'training-plan-status',
        }


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
