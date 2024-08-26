import copy
from json import decoder
import os
import tempfile
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch, ANY
from fedbiomed.common.optimizers.generic_optimizers import DeclearnOptimizer
from fedbiomed.common.serializer import Serializer


#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

# import dummy classes
from testsupport.fake_uuid import FakeUuid
from testsupport.fake_message import FakeMessages
from testsupport.fake_secagg_manager import FakeSecaggServkeyManager, FakeSecaggBiprimeManager
from testsupport import fake_training_plan

import torch
from fedbiomed.common.optimizers.declearn import YogiModule, ScaffoldClientModule, RidgeRegularizer

from fedbiomed.node.environ import environ
from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes, _BaseEnum, TrainingPlans, __messaging_protocol_version__
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.message import NodeMessages, TrainRequest, SecaggReply, SecaggDeleteReply
from fedbiomed.common.models import TorchModel
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.node import Node
from fedbiomed.node.round import Round
from fedbiomed.node.dataset_manager import DatasetManager


class TestNode(NodeTestCase):

    @classmethod
    def setUpClass(cls):
        # Important to instantiate fake environ
        super().setUpClass()

        # --------------------------------------

        # defining common side effect functions
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeMessages(msg)
            return fake_node_msg

        cls.node_msg_side_effect = node_msg_side_effect

    def setUp(self):
        """Sets up objects for unit tests"""

        self.database_val = [
            {'database_id': '1234',
             'path': '/path/to/my/dataset',
             'name': 'test_dataset'}
        ]
        self.database_list = [
            {'database_id': '1234',
             'path': '/path/to/my/dataset',
             'name': 'test_dataset1'},
            {'database_id': '5678',
             'path': '/path/to/another/dataset',
             'name': 'test_dataset2'}
        ]

        self.database_id = {
            'database_id': '1234',
            'path': '/path/to/my/dataset',
            'name': 'test_dataset1'
        }

        # patchers
        self.grpc_controller_patch = patch('fedbiomed.transport.controller.GrpcController.__init__',
                                           autospec=True,
                                           return_value=None)
        self.grpc_send_patch = patch('fedbiomed.transport.controller.GrpcController.send', autospec=True)

        self.grpc_controller_patcher = self.grpc_controller_patch.start()
        self.grpc_send_mock = self.grpc_send_patch.start()


        self.task_queue_patch = patch('fedbiomed.common.tasks_queue.TasksQueue.__init__',
                                      autospec=True,
                                      return_value=None)
        self.task_patcher = self.task_queue_patch.start()

        self.exchange_patch = patch('fedbiomed.node.node.EventWaitExchange', autospec=True)
        self.exchange_patcher = self.exchange_patch.start()

        self.n2n_router_patch = patch('fedbiomed.node.node.NodeToNodeRouter', autospec=True)
        self.n2n_router_patcher = self.n2n_router_patch.start()

        # mocks
        mock_dataset_manager = DatasetManager()
        mock_dataset_manager.search_by_tags = MagicMock(return_value=self.database_val)
        mock_dataset_manager.list_my_data = MagicMock(return_value=self.database_list)
        mock_model_manager = MagicMock()
        mock_dataset_manager.reply_training_plan_status_request = MagicMock(return_value=None)
        mock_dataset_manager.get_by_id = MagicMock(return_value=self.database_id)

        self.model_manager_mock = mock_model_manager

        # creating Node objects
        self.n1 = Node(mock_dataset_manager, mock_model_manager)
        self.n2 = Node(mock_dataset_manager, mock_model_manager)

    def tearDown(self) -> None:
        # stopping patches
        self.grpc_send_patch.stop()
        self.task_queue_patch.stop()
        self.grpc_controller_patch.stop()
        self.exchange_patch.stop()
        self.n2n_router_patch.stop()

    @patch('fedbiomed.common.tasks_queue.TasksQueue.add')
    def test_node_01_add_task_normal_case_scenario(self, task_queue_add_patcher):
        """Tests add_task method (in the normal case scenario)"""

        for command in ['train', 'secagg']:
            # arguments
            # a dummy message
            node_msg_request_create_task = {
                'msg': "a message for testing",
                'command': command
            }
            # action
            self.n1.add_task(node_msg_request_create_task)

            # checks
            task_queue_add_patcher.assert_called_once_with(node_msg_request_create_task)
            task_queue_add_patcher.reset_mock()

    @patch('fedbiomed.node.secagg._secagg_setups.BPrimeManager')
    @patch('fedbiomed.node.secagg._secagg_setups.SKManager')
    @patch('fedbiomed.node.node.Node.add_task')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    def test_node_02_on_message_normal_case_scenario_train_secagg_reply(
            self,
            node_msg_req_create_patcher,
            node_add_task_patcher,
            patch_servkey_manager,
            patch_biprime_manager
    ):
        """Tests `on_message` method (normal case scenario), with train/secagg command"""
        # test 1: test normal case scenario, where `command` = 'train' or 'secagg'

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        node_msg_req_create_patcher.side_effect = TestNode.node_msg_side_effect
        for command in ['train', 'secagg']:
            train_msg = {
                'command': command
            }
            # action
            self.n1.on_message(train_msg)

            # checks
            node_add_task_patcher.assert_called_once()
            node_add_task_patcher.reset_mock()

    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    def test_node_03_on_message_normal_case_scenario_ping(
            self,
            node_msg_request_patch,
            node_msg_reply_patch,
    ):
        """Tests `on_message` method (normal case scenario), with ping command"""
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect

        # defining arguments
        ping_msg = {
            'command': 'ping',
            'researcher_id': 'researcher_id_1234',
        }

        # action
        self.n1.on_message(ping_msg)
        ping_msg.update(
            {
                'node_id': environ['NODE_ID'],
                'command': 'pong',
                'success': True
            })
        # checks
        self.grpc_send_mock.assert_called_once()

    @patch('fedbiomed.node.node.SecaggManager')
    def test_node_04_on_message_normal_case_scenario_secagg_delete(
            self,
            skm,
    ):
        """Tests `on_message` method (normal case scenario), with secagg-delete command"""

        # defining arguments
        secagg_delete = {
            "protocol_version": str(__messaging_protocol_version__),
            'command': 'secagg-delete',
            'researcher_id': 'researcher_id_1234',
            'secagg_id': 'my_test_secagg_id',
            'element': 0,
            'experiment_id': 'a_dummy_experiment_id',
        }

        skm.return_value.return_value.remove.return_value = True

        # action
        self.n1.on_message(secagg_delete)

        secagg_delete_reply = copy.deepcopy(secagg_delete)
        secagg_delete_reply.update(
            {
                'node_id': environ['NODE_ID'],
                'success': True,
                'msg': 'Delete request is successful'
            })
        del secagg_delete_reply['experiment_id']
        del secagg_delete_reply['element']
        # checks
        self.grpc_send_mock.assert_called_once()

    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    def test_node_05_on_message_normal_case_scenario_search(self,
                                                            node_msg_request_patch,
                                                            node_msg_reply_patch
                                                            ):
        """Tests `on_message` method (normal case scenario), with search command"""
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect

        # defining arguments
        search_msg = {
            'command': 'search',
            'researcher_id': 'researcher_id_1234',
            'tags': ['#some_tags']
        }
        # action
        self.n1.on_message(search_msg)

        # argument `search_msg` will be modified: we will check if
        # message has been modified accordingly

        self.database_val[0].pop('path', None)
        search_msg.pop('tags', None)
        search_msg.update({'success': True,
                           'node_id': environ['NODE_ID'],
                           'databases': self.database_val,
                           'count': len(self.database_val)})
        self.grpc_send_mock.assert_called_once()

    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    def test_node_06_on_message_normal_case_scenario_list(self,
                                                          node_msg_request_patch,
                                                          node_msg_reply_patch
                                                          ):
        """Tests `on_message` method (normal case scenario), with list command"""
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        # defining arguments
        list_msg = {
            'command': 'list',
            'researcher_id': 'researcher_id_1234',
        }

        # action
        self.n1.on_message(list_msg)

        # updating `list_msg` value to match the one sent through
        # Messaging class (here we are removing `path` and `dataset_id`
        # entries in the `database_list`)
        for d in self.database_list:
            for key in ['path', 'dataset_id']:
                d.pop(key, None)

        list_msg.update({
            'success': True,
            'node_id': environ['NODE_ID'],
            'databases': self.database_list,
            'count': len(self.database_list)})

        # checks
        self.grpc_send_mock.assert_called_once()

    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    def test_node_07_on_message_normal_case_scenario_model_status(self,
                                                                  node_msg_request_patch,
                                                                  ):
        """Tests normal case scenario, if command is equals to 'training-plan-status"""
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        # defining arguments
        model_status_msg = {
            'command': 'training-plan-status',
            'researcher_id': 'researcher_id_1234',
        }

        # action
        self.n1.on_message(model_status_msg)

        # checks
        self.model_manager_mock.reply_training_plan_status_request.assert_called_once_with(model_status_msg)

    @patch('fedbiomed.node.node.Node.send_error')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    def test_node_08_on_message_unknown_command(self,
                                                node_msg_request_patch,
                                                send_err_patch):
        """Tests Exception is handled if command is not a known command
        (in `on_message` method)"""
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect

        # defining arguments
        unknown_cmd = 'unknown'
        researcher_id = 'researcher_id_1234'
        unknown_cmd_msg = {
            'command': unknown_cmd,
            'researcher_id': researcher_id,
        }

        # action
        self.n1.on_message(unknown_cmd_msg)

        # check
        send_err_patch.assert_called_once_with(ErrorNumbers.FB301,
                                               extra_msg=f"Command `{unknown_cmd}` is not implemented",
                                               researcher_id='researcher_id_1234')


    @patch('fedbiomed.node.node.Node.send_error')
    def test_node_10_on_message_fail_getting_msg_field(self,
                                                       msg_send_error_patch):
        """Tests case where a KeyError (unable to extract fields of `msg`) Exception
        is raised during process"""
        resid = 'researcher_id_1234'
        no_command_msg = {
            'researcher_id': resid,
        }

        # action
        self.n1.on_message(no_command_msg)

        # check
        msg_send_error_patch.assert_called_once_with(ErrorNumbers.FB301,
                                                     extra_msg="'command' property was not found",
                                                     researcher_id=resid)

    @patch('fedbiomed.node.node.Node.send_error')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    def test_node_11_on_message_fail_msg_not_serializable(self,
                                                          node_msg_request_patch,
                                                          node_msg_reply_patch,
                                                          msg_send_error_patch):
        """Tests case where a TypError is raised (because unable to serialize message)"""

        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        self.grpc_send_mock.side_effect = TypeError

        # defining arguments
        command = 'ping'
        resid = 'researcher_id_1234'
        ping_msg = {
            'command': command,
            'researcher_id': resid,
            'request_id': '1234'
        }

        # action
        self.n1.on_message(ping_msg)

        # checks
        # check if `Messaging.send_message` has been called with good arguments
        msg_send_error_patch.assert_called_once_with(ErrorNumbers.FB301,
                                                     extra_msg='Message was not serializable',
                                                     researcher_id=resid)

    @patch('fedbiomed.node.node.Round', autospec=True)
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    def test_node_12_parser_task_train_create_round(self,
                                                    node_msg_request_patch,
                                                    history_monitor_patch,
                                                    round_patch,

                                                    ):
        """Tests if rounds are created accordingly - running normal case scenario
        (in `parser_task_train` method)"""

        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None
        round_patch.return_value.initialize_arguments.return_value = None

        # test 1: case where 1 dataset has been found
        dict_msg_1_dataset = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training': True,
            'training_plan': 'dummy_plan',
            'training_plan_class': 'my_test_training_plan',
            'experiment_id': 'experiment_id_1234',
            'researcher_id': 'researcher_id_1234',
            'dataset_id': 'dataset_id_1234',
        }
        msg_1_dataset = NodeMessages.format_incoming_message(dict_msg_1_dataset)

        round = self.n1.parser_task_train(msg_1_dataset)

        # checks
        self.assertIsInstance(round, Round)
        round_patch.assert_called_once()


    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    @patch('fedbiomed.node.round.Round.__init__')
    def test_node_13_parser_task_train_no_dataset_found(self,
                                                        round_init,
                                                        node_msg_request_patch,
                                                        history_monitor_patch,
                                                        node_msg_reply_patch,
                                                        ):
        """Tests parser_task_train method, case where no dataset has been found """
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        history_monitor_patch.return_value = None
        round_init.return_value = None

        # defining arguments
        resid = 'researcher_id_1234'
        msg_without_datasets = TrainRequest(**{
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training_plan': 'TP',
            'training_plan_class': 'my_test_training_plan',
            'params': {"x": 0},
            'experiment_id': 'experiment_id_1234',
            'researcher_id': resid,
            'dataset_id': 'dataset_id_1234',
            'request_id': 'request-id',
            'aggregator_args': {},
            'state_id': 'state',
            'training': True,
            'command': 'train',
            'round': 1
        })
        # create tested object

        mock_dataset_manager = MagicMock()
        # return emtpy list to mimic dataset that havenot been found
        mock_dataset_manager.get_by_id = MagicMock(return_value=None)

        self.n1.dataset_manager = mock_dataset_manager

        # action

        self.n1.parser_task_train(msg_without_datasets)

        # checks
        self.grpc_send_mock.assert_called_once()

    @patch('fedbiomed.node.node.Round', autospec=True)
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    def test_node_14_parser_task_train_create_round_deserializer_str_msg(self,
                                                                         history_monitor_patch,
                                                                         round_patch
                                                                         ):
        """Tests if message is correctly deserialized if message is in string"""

        # defining arguments
        dict_msg_1_dataset = {
            "protocol_version": str(__messaging_protocol_version__),
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training_plan': 'TP',
            'training_plan_class': 'my_test_training_plan',
            'params': {"x": 0},
            'experiment_id': 'experiment_id_1234',
            'state_id': None,
            'secagg_arguments': {
                "secagg_biprime_id": None,
                "secagg_servkey_id": None,
                "secagg_random": None,
                "secagg_clipping_range": None
            },
            "round": 1,
            'researcher_id': 'researcher_id_1234',
            'command': 'train',
            'dataset_id': 'dataset_id_1234',
            'training': True,
            'aggregator_args': {},
            "aux_vars": ["url_shared_aux_var", "url_bynode_aux_var"],
        }
        # we convert this dataset into a string
        msg1_dataset = NodeMessages.format_incoming_message(dict_msg_1_dataset)
        round_patch.return_value.initialize_arguments.return_value = None

        # defining patchers

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None

        # action
        self.n1.parser_task_train(msg1_dataset)

        # checks
        round_patch.assert_called_once_with(
            model_kwargs=dict_msg_1_dataset['model_args'],
            training_kwargs=dict_msg_1_dataset['training_args'],
            training=True,
            dataset=self.database_id,
            params=dict_msg_1_dataset['params'],
            experiment_id=dict_msg_1_dataset['experiment_id'],
            researcher_id=dict_msg_1_dataset['researcher_id'],
            history_monitor=unittest.mock.ANY,
            aggregator_args=None,
            node_args=None,
            training_plan=dict_msg_1_dataset['training_plan'],
            training_plan_class=dict_msg_1_dataset['training_plan_class'],
            round_number=1,
            dlp_and_loading_block_metadata=None,
            aux_vars= dict_msg_1_dataset['aux_vars']
        )

    @patch('fedbiomed.node.node.Round', autospec=True)
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    def test_node_15_parser_task_train_create_round_deserializer_bytes_msg(self,
                                                                           history_monitor_patch,
                                                                           round_patch
                                                                           ):
        """Tests if message is correctly deserialized if message is in bytes"""

        # defining arguments
        dict_msg_1_dataset = {
            "protocol_version": str(__messaging_protocol_version__),
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training": True,
            "training_plan": "TP",
            "training_plan_class": "my_test_training_plan",
            "params": {"x": 0},
            "experiment_id": "experiment_id_1234",
            "state_id": None,
            "researcher_id": "researcher_id_1234",
            "secagg_arguments": None,
            "command": "train",
            "dataset_id": "dataset_id_1234",
            'aggregator_args': {},
            "aux_vars": ["single_url_aux_var"],
            "round": 0
        }

        #
        msg_1_dataset = NodeMessages.format_incoming_message(dict_msg_1_dataset)

        # defining patchers

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None
        round_patch.return_value.initialize_arguments.return_value = None

        # action
        self.n1.parser_task_train(msg_1_dataset)

        # checks
        round_patch.assert_called_once_with(
            model_kwargs=dict_msg_1_dataset['model_args'],
            training_kwargs=dict_msg_1_dataset['training_args'],
            training=True,
            dataset=self.database_id,
            params=dict_msg_1_dataset['params'],
            experiment_id=dict_msg_1_dataset['experiment_id'],
            researcher_id=dict_msg_1_dataset['researcher_id'],
            history_monitor=unittest.mock.ANY,
            aggregator_args=None,
            node_args=None,
            training_plan=dict_msg_1_dataset['training_plan'],
            training_plan_class=dict_msg_1_dataset['training_plan_class'],
            round_number=0,
            dlp_and_loading_block_metadata=None,
            aux_vars= dict_msg_1_dataset['aux_vars']
        )


    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_16_task_manager_exception_raised_task_queue(self,
                                                              tasks_queue_get_patch):
        """Simulates an Exception (SystemError) triggered by `tasks_queue.get`
        """
        # defining patchers
        tasks_queue_get_patch.side_effect = SystemError("mimicking an exception coming from ")

        # action
        with self.assertRaises(SystemError):
            # checks if `SystemError` is caught (triggered by patched `tasks_queue.get`)
            self.n1.task_manager()

    # NOTA BENE: for test 19 to test 24 (testing `task_manager` method)
    # Since we don't have any proper way to stop the infinite loop defined
    # in the method, we are triggering `SystemExit` Exception to leave it
    # (SystemExit is an exception that is not caught by statement
    # `except Exception as e:`). When a more graceful way of exiting infinite loop
    # will be created, those tests should be updated


    @patch('fedbiomed.node.node.NodeMessages.format_outgoing_message')
    @patch('fedbiomed.node.node.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    def test_node_17_task_manager_exception_raised(self,
                                                   task_done_mock,
                                                   tasks_queue_get_patch,
                                                   reply_create_patch,
                                                   request_create_patch):
        """Tests case where `NodeMessages.format_outgoing_message` method raises an exception
        and then the format_incoming_message raises another exception(SystemExit).
        """
        # defining patchers
        tasks_queue_get_patch.return_value = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training": True,
            "training_plan": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params": {"x": 0},
            "experiment_id": "experiment_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "secagg_id": None,
            "secagg_random": None,
            "secagg_clipping_range": None,
            "round": 1,
            "dataset_id": "dataset_id_1234"
        }
        request_create_patch.side_effect = Exception
        reply_create_patch.side_effect = SystemExit(
            "mimicking an exception" + " coming from NodeMessages.format_outgoing_message")  # noqa
        self.grpc_send_mock.return_value = None

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is caught
            # (should be triggered by `Node.parser_task_train` method)
            self.n1.task_manager()

    @patch('fedbiomed.node.node.Node.parser_task_train')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    def test_node_18_task_manager_train_exception_raised_send_message(self,
                                                                      tasks_queue_done_patch,
                                                                      tasks_queue_get_patch,
                                                                      node_parser_task_train_patch):
        """Tests case where `messaging.send_message` method
        raises an exception (SystemExit).
        """
        # defining patchers
        tasks_queue_get_patch.return_value = {}

        tasks_queue_get_patch.return_value = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "aggregator_args": {},
            "training": True,
            "training_plan": "TP",
            "training_plan_class": "my_test_training_plan",
            "params": {"x": 0},
            "experiment_id": "experiment_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "secagg_id": None,
            "secagg_random": None,
            "secagg_clipping_range": None,
            "round": 1,
            "dataset_id": "dataset_id_1234"
        }
        # defining arguments and attributes
        self.grpc_send_mock.side_effect = SystemExit("Mimicking an exception happening in" + "`send_message` method")  # noqa

        Round = MagicMock()
        Round.run_model_training = MagicMock(run_model_training=None)
        node_parser_task_train_patch.return_value = Round

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is caught (should be triggered by
            # `messaging.send_message` method )
            self.n1.task_manager()

    @patch('fedbiomed.node.secagg._secagg_setups.BPrimeManager')
    @patch('fedbiomed.node.secagg._secagg_setups.SKManager')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.node.node.Node._task_secagg')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_19_task_manager_secagg_exception_raised_task_done(
            self,
            tasks_queue_get_patch,
            task_secagg_patch,
            tasks_queue_task_done_patch,
            patch_servkey_manager,
            patch_biprime_manager):
        """Tests if an Exception (SystemExit) is triggered when calling
        `TasksQueue.task_done` method for secagg message"""
        # defining patchers
        tasks_queue_get_patch.return_value = {
            "protocol_version": '99.99',
            "researcher_id": "my_test_researcher",
            "secagg_id": "my_test_secagg",
            "element": 33,
            "experiment_id": "my_experiment",
            "parties": [],
            "command": "secagg"
        }
        task_secagg_patch.return_value = None
        self.grpc_send_mock.return_value = None

        tasks_queue_task_done_patch.side_effect = SystemExit(
            "Mimicking an exception happening in" + "`TasksQueue.task_done` method")  # noqa

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is raised (should be triggered by `TasksQueue.task_done`)
            self.n1.task_manager()

        # check that `Messaging.send_message` has not been called
        self.assertEqual(self.grpc_send_mock.call_count, 0)

    @patch('fedbiomed.transport.controller.GrpcController.start')
    def test_node_20_start_messaging_normal_case_scenario(self,
                                                          msg_start_patch):
        """Tests `start_messaging` method (normal_case_scenario)"""
        # arguments
        block = True

        # action
        self.n1.start_messaging(block)

        # checks
        msg_start_patch.assert_called_once_with(block)

    def test_node_21_send_error_normal_case_scenario(self):
        """Tests `send_error` method (normal case scenario)"""
        # arguments
        errnum = ErrorNumbers.FB100
        extra_msg = "this is a test_send_error"
        researcher_id = 'researcher_id_1224'

        # action
        self.n1.send_error(errnum, extra_msg, researcher_id)

        # checks
        self.grpc_send_mock.assert_called_once()

    def test_node_22_on_message_search_privacy_obfuscation(self):
        """Tests that privacy-sensitive information is not revealed (here path files)"""

        databases = [dict(
            data_type='medical-folder',
            dataset_parameters={'tabular_file': 'path/to/tabular/file',
                                'index_col': 0},
            **self.database_val[0]
        )]

        dataset_manager = DatasetManager()
        dataset_manager.search_by_tags = MagicMock(return_value=databases)
        n3 = Node(dataset_manager, self.model_manager_mock)

        search_msg = {
            "protocol_version": str(__messaging_protocol_version__),
            'command': 'search',
            'researcher_id': 'researcher_id_1234',
            'request_id': 'request_id',
            'tags': ['#some_tags']
        }
        # action
        n3.on_message(search_msg)

        # check privacy-sensitive info a case-by-case basis
        database_info = self.grpc_send_mock.call_args[0][1].get_param('databases')[0]
        self.assertNotIn('path', database_info)
        self.assertNotIn('tabular_file', database_info['dataset_parameters'])

    def test_node_23_task_secagg(
        self
    ):
        """Tests `_task_secagg` normal (successful) case"""

        req = {"protocol_version": str(__messaging_protocol_version__),
               'researcher_id': 'party1',
               'request_id': 'request',
               'secagg_id': 'my_dummy_secagg_id',
               'element': 0,
               'experiment_id': 'my_test_experiment',
               'parties': ['party1', 'party2', 'party3'],
               'command': 'secagg'}
        # Create request
        request = NodeMessages.format_incoming_message(req)

        # Test .setup()execution. It is normal the get result as success False since setup will fail
        # due to not existing certificate files

        with patch('fedbiomed.node.node.GrpcController.send') as grpc_send:
            self.n1._task_secagg(request)

        grpc_send.assert_called_once_with(
            SecaggReply(**{'researcher_id': req['researcher_id'],
                           'protocol_version': str(__messaging_protocol_version__),
                           'secagg_id': req['secagg_id'],
                           'request_id': 'request',
                           'success': False,
                           'node_id': environ["ID"],
                           'msg': f'Can not setup secure aggregation context on node for {req["secagg_id"]}.',
                           'command': 'secagg'})
        )


        # Test setup error case ---------------------------------------------------------------
        req["element"] = 12
        request = NodeMessages.format_incoming_message(req)
        with patch('fedbiomed.node.node.GrpcController.send') as grpc_send:
            self.n1._task_secagg(request)
        grpc_send.assert_called_once_with(
            SecaggReply(**{'researcher_id': req['researcher_id'],
                           'protocol_version': str(__messaging_protocol_version__),
                           'secagg_id': req['secagg_id'],
                           'request_id': 'request',
                           'success': False,
                           'node_id': environ["ID"],
                           'msg': f"FB318: Secure aggregation setup error: Received bad request message: incorrect `element` {req['element']}",
                           'command': 'secagg'})
        )


    def test_node_24_task_secagg_delete(
            self
    ):
        """Tests `_task_secagg` with bad message values"""

        # Bad element type --------------------------------------------------------------------------
        req = {"protocol_version": str(__messaging_protocol_version__),
               'researcher_id': 'party1',
               'secagg_id': 'my_dummy_secagg_id',
               'request_id': 'request',
               'element': 11,
               'experiment_id': 'my_test_experiment',
               'command': 'secagg-delete'}
        # Create request
        request = NodeMessages.format_incoming_message(req)
        self.n1._task_secagg_delete(request)

        self.grpc_send_mock.assert_called_once_with(
            unittest.mock.ANY,
            SecaggDeleteReply(
                **{'protocol_version': req['protocol_version'],
                   'researcher_id': req['researcher_id'],
                   'secagg_id': req['secagg_id'],
                   'request_id': 'request',
                   'success': False,
                   'node_id': environ["ID"],
                   'msg': 'FB321: Secure aggregation delete error: Can not instantiate SecaggManager object FB318: '
                          f'Secure aggregation setup error: received bad message: incorrect `element` {req["element"]}',
                   'command': 'secagg-delete'}))
        self.grpc_send_mock.reset_mock()

        #
        # Test remove status ----------------------------------------------------------------
        # status will be false since there is no registry in DB
        req["element"] = 0
        request = NodeMessages.format_incoming_message(req)
        self.n1._task_secagg_delete(request)
        self.grpc_send_mock.assert_called_once_with(
            unittest.mock.ANY,
            SecaggDeleteReply(
                **{'researcher_id': req['researcher_id'],
                   'protocol_version': str(__messaging_protocol_version__),
                   'secagg_id': req['secagg_id'],
                   'request_id': 'request',
                   'success': False,
                   'node_id': environ["ID"],
                   'msg': 'FB321: Secure aggregation delete error: no such secagg context element in node database for '
                          f'node_id={environ["ID"]} secagg_id=my_dummy_secagg_id',
                   'command': 'secagg-delete'}))
        self.grpc_send_mock.reset_mock()

        # # Test raising error
        with patch("fedbiomed.node.node.SecaggManager") as skm:
            skm.return_value.return_value.remove.side_effect = Exception
            req["element"] = 0
            request = NodeMessages.format_incoming_message(req)
            self.n1._task_secagg_delete(request)
            self.grpc_send_mock.assert_called_once_with(
                unittest.mock.ANY,
                SecaggDeleteReply(
                    **{ 'researcher_id': req['researcher_id'],
                        'protocol_version': str(__messaging_protocol_version__),
                        'secagg_id': req['secagg_id'],
                        'request_id': 'request',
                        'success': False,
                        'node_id': environ["ID"],
                        'msg': 'FB321: Secure aggregation delete error: error during secagg delete on '
                        f'node_id={environ["ID"]} secagg_id={req["secagg_id"]}: ',
                        'command': 'secagg-delete'}))

    def test_node_31_reply(
            self
    ):

        # Test faulty message data ---------------------------------------------
        self.n1.reply({
            'faulty_message': "faulty_value"
        })

        self.grpc_send_mock.assert_called_once()
        self.grpc_send_mock.reset_mock()

        # Test fualty type of message -----------------------------------------------
        self.n1.reply({
            'faulty_type'
        })
        self.grpc_send_mock.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
