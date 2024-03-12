import os
import shutil
import unittest
from unittest.mock import MagicMock, create_autospec, patch

from testsupport.base_case import ResearcherTestCase  # Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_mocks import MockRequestModule
from testsupport.fake_training_plan import FakeTorchTrainingPlan
from testsupport import fake_training_plan

from fedbiomed.common.message import TrainReply
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.federated_workflows.jobs import Job, TrainingJob


class TestJob(ResearcherTestCase, MockRequestModule):
    """Tests Job class and all of its subclasses"""
    def setUp(self):
        MockRequestModule.setUp(self, module="fedbiomed.researcher.federated_workflows.jobs._job.Requests")
        self.ic_from_file_patch = patch('fedbiomed.researcher.federated_workflows.jobs._training_job.utils.import_class_object_from_file', autospec=True)
        self.ic_from_file_mock = self.ic_from_file_patch.start()
        self.patch_serializer = patch("fedbiomed.common.serializer.Serializer")
        self.mock_serializer = self.patch_serializer.start()

        # Globally create mock for Model and FederatedDataset
        self.model = create_autospec(BaseTrainingPlan, instance=False)
        self.fds = MagicMock(spec=FederatedDataSet)
        self.fds.data = MagicMock(return_value={})
        self.model = FakeTorchTrainingPlan
        self.model.save_code = MagicMock()
        self.ic_from_file_mock.return_value = (fake_training_plan, MagicMock(spec=BaseTrainingPlan) )

    def tearDown(self) -> None:

        #self.patcher4.stop()
        self.ic_from_file_patch.stop()
        self.patch_serializer.stop()

        # Remove if there is dummy model file
        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

        super().tearDown()

    def test_job_01_base_job(self):
        # Job should be default-constructible
        job = Job()
        self.assertIsNotNone(job._keep_files_dir)  # must be initialized by Job
        self.assertTrue(isinstance(job.nodes, list) and len(job.nodes) == 0)  # nodes must be empty list by default
        # Job can take nodes and keep_files_dir as arguments
        mynodes = ['first-node', 'second-node']
        job = Job(
            nodes = mynodes,
            keep_files_dir='keep_files_dir'
        )
        self.assertEqual(job._keep_files_dir, 'keep_files_dir')
        self.assertTrue(all(x == y for x,y in zip(job.nodes, mynodes)))
        # Nodes can be set a posteriori
        job.nodes = ['another-node']
        self.assertTrue(all(x == y for x,y in zip(job.nodes, ['another-node'])))

    @patch('fedbiomed.researcher.federated_workflows._training_plan_workflow.uuid.uuid4', return_value='UUID')
    def test_job_02_training_job(self, mock_uuid):
        # TrainingJob should be default-constructible
        job = TrainingJob()

        # Initializing a training plan instance via Job must call:
        # 1) the training plan's default constructor
        # 2) training plan's post init
        mock_tp_class = MagicMock()
        mock_tp_class.return_value = MagicMock(spec=BaseTrainingPlan)
        mock_tp_class.__name__ = 'mock_tp_class'

        mock_tp = mock_tp_class()

        # Calling start_nodes_training_round must:
        # 1) call the `Requests.send` function to initiate training on the nodes
        # 2) return the properly formatted replies
        job.nodes = ['alice', 'bob']
        self.fds.data = MagicMock(return_value={
            'alice': {'dataset_id': 'alice_data'},
            'bob': {'dataset_id': 'bob_data'},
        })
        fake_node_state_ids = {
            'alice': 'alide_nsid',
            'bob': 'bob_nsid'
        }
        mock_tp.get_model_params.return_value = MagicMock(spec=dict)
        mock_tp.source.return_value = MagicMock(spec=str)
        self.mock_federated_request.errors.return_value = {}
        self.mock_federated_request.replies.return_value ={
            'alice': TrainReply(**self._get_train_reply('alice', self.fds.data()['alice']['dataset_id'])),
            'bob': TrainReply(**self._get_train_reply('bob', self.fds.data()['bob']['dataset_id'])),
        }
        with patch("time.perf_counter") as mock_perf_counter:
            mock_perf_counter.return_value = 0
            replies = job.start_nodes_training_round(
                job_id='some_id',
                round_=1,
                training_plan=mock_tp,
                training_args=TrainingArgs({}, only_required=False),
                model_args=None,
                data=self.fds,  # mocked FederatedDataSet class
                nodes_state_ids=fake_node_state_ids,
                aggregator_args={},
                optim_aux_var={
                    'shared': {},
                    'node-specific': {
                        'alice':'node-specific',
                        'bob':'node-specific'
                    }
                }
            )
        # The `send` function of the Requests module is always only called
        # once regardless of the number of nodes
        self.maxDiff = None
        self.mock_requests.return_value.send.called_once_with(
            [
                (
                    {'alice': self._get_msg(
                        mock_tp, {}, 'alice', fake_node_state_ids, self.fds.data()
                        ),
                     'bob': self._get_msg(mock_tp, {}, 'bob', fake_node_state_ids, self.fds.data())},
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
        self.assertDictEqual(replies, expected_replies)

        # test extract_received_optimizer_aux_var_from_round
        extracted_aux_var = job.extract_received_optimizer_aux_var_from_round(
            round_id=42,
            training_replies={
                42: {
                    'alice': {
                        'optim_aux_var': {'module': 'params'}
                    },
                    'bob': {
                        'optim_aux_var': {'module': 'params'}
                    }
                }
            }
        )
        self.assertDictEqual(extracted_aux_var, {'module': {'alice': 'params', 'bob': 'params'}})

    def _get_msg(self,
                 mock_tp,
                 secagg_arguments,
                 node_id,
                 state_ids,
                 data):
        return {
        'researcher_id': environ['RESEARCHER_ID'],
        'job_id': 'some_id',
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
                         dataset_id):
        return {
            'researcher_id': environ['RESEARCHER_ID'],
            'job_id': 'some_id',
            'success': True,
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
            'optim_aux_var': None,
            'encryption_factor': None,
        }

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
