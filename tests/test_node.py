import copy
from json import decoder
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch


#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

# import dummy classes
from testsupport.fake_message import FakeMessages
from testsupport.fake_node_secagg import FakeSecaggServkeySetup, FakeSecaggBiprimeSetup
from testsupport.fake_secagg_manager import FakeSecaggServkeyManager, FakeSecaggBiprimeManager

from fedbiomed.node.environ import environ
from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes, _BaseEnum
from fedbiomed.common.message import NodeMessages
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
        self.task_queue_patch = patch('fedbiomed.common.messaging.Messaging.__init__',
                                      autospec=True,
                                      return_value=None)
        self.task_patcher = self.task_queue_patch.start()

        self.messaging_patch = patch('fedbiomed.common.tasks_queue.TasksQueue.__init__',
                                     autospec=True,
                                     return_value=None)
        self.messaging_patcher = self.messaging_patch.start()

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
        self.task_queue_patch.stop()
        self.messaging_patch.stop()

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

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.node.node.Node.add_task')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
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
            node_add_task_patcher.assert_called_once_with(train_msg)
            node_add_task_patcher.reset_mock()

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_03_on_message_normal_case_scenario_ping(
            self,
            node_msg_request_patch,
            node_msg_reply_patch,
            messaging_send_msg_patch
    ):
        """Tests `on_message` method (normal case scenario), with ping command"""
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect

        # defining arguments
        ping_msg = {
            'command': 'ping',
            'researcher_id': 'researcher_id_1234',
            'sequence': 1234
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
        messaging_send_msg_patch.assert_called_once_with(ping_msg)

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.node.node.SecaggBiprimeManager.remove')
    @patch('fedbiomed.node.node.SecaggServkeyManager.remove')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_04_on_message_normal_case_scenario_secagg_delete(
            self,
            node_msg_request_patch,
            node_msg_reply_patch,
            messaging_send_msg_patch,
            secagg_servkey_manager_remove_patch,
            secagg_biprime_manager_remove_patch,
            patch_servkey_manager,
            patch_biprime_manager,
    ):
        """Tests `on_message` method (normal case scenario), with secagg-delete command"""
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        secagg_servkey_manager_remove_patch.return_value = True
        secagg_biprime_manager_remove_patch.return_value = True

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # defining arguments
        secagg_delete = {
            'command': 'secagg-delete',
            'researcher_id': 'researcher_id_1234',
            'secagg_id': 'my_test_secagg_id',
            'sequence': 1234,
            'element': 0,
            'job_id': 'a_dummy_job_id',
        }

        # action
        self.n1.on_message(secagg_delete)
        secagg_delete_reply = copy.deepcopy(secagg_delete)
        secagg_delete_reply.update(
            {
                'node_id': environ['NODE_ID'],
                'success': True,
                'msg': ''
            })
        del secagg_delete_reply['job_id']
        del secagg_delete_reply['element']
        # checks
        messaging_send_msg_patch.assert_called_once_with(secagg_delete_reply)

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_05_on_message_normal_case_scenario_search(self,
                                                            node_msg_request_patch,
                                                            node_msg_reply_patch,
                                                            messaging_send_msg_patch
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
        messaging_send_msg_patch.assert_called_once_with(search_msg)

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_06_on_message_normal_case_scenario_list(self,
                                                          node_msg_request_patch,
                                                          node_msg_reply_patch,
                                                          messaging_send_msg_patch
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
        messaging_send_msg_patch.assert_called_once_with(list_msg)

    @patch('fedbiomed.common.message.NodeMessages.request_create')
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
        self.model_manager_mock.reply_training_plan_status_request.assert_called_once_with(model_status_msg,
                                                                                   self.n1.messaging)

    @patch('fedbiomed.node.node.Node.send_error')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
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
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_09_on_message_fail_reading_json(self,
                                                  node_msg_request_patch,
                                                  node_msg_reply_patch,
                                                  messaging_patch,
                                                  msg_send_error_patch):
        """Tests case where a JSONDecodeError is triggered (JSON cannot be created)"""

        def messaging_side_effect(*args, **kwargs):
            raise decoder.JSONDecodeError('mimicking a JSONDEcodeError',
                                          doc='a_json_doc', pos=1)

        # JSONDecodeError can be raised from messaging class
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        messaging_patch.side_effect = messaging_side_effect

        # defining arguments
        command = 'ping'
        resid = 'researcher_id_1234'
        ping_msg = {
            'command': command,
            'researcher_id': resid,
            'sequence': 1234
        }

        # action
        self.n1.on_message(ping_msg)

        # check
        msg_send_error_patch.assert_called_once_with(ErrorNumbers.FB301,
                                                     extra_msg="Not able to deserialize the message",
                                                     researcher_id=resid)

    @patch('fedbiomed.node.node.Node.send_error')
    def test_node_10_on_message_fail_getting_msg_field(self,
                                                       msg_send_error_patch):
        """Tests case where a KeyError (unable to extract fields of `msg`) Exception
        is raised during process"""
        resid = 'researcher_id_1234'
        no_command_msg = {
            'researcher_id': resid,
            'sequence': 1234
        }

        # action
        self.n1.on_message(no_command_msg)

        # check
        msg_send_error_patch.assert_called_once_with(ErrorNumbers.FB301,
                                                     extra_msg="'command' property was not found",
                                                     researcher_id=resid)

    @patch('fedbiomed.node.node.Node.send_error')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_11_on_message_fail_msg_not_serializable(self,
                                                          node_msg_request_patch,
                                                          node_msg_reply_patch,
                                                          messaging_patch,
                                                          msg_send_error_patch):
        """Tests case where a TypError is raised (because unable to serialize message)"""

        # a TypeError can be raised from json serializer (ie from  `Messaging.send_message`)
        def messaging_side_effect(*args, **kwargs):
            raise TypeError('Mimicking a TypeError happening when serializing message')

        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        messaging_patch.side_effect = messaging_side_effect

        # defining arguments
        command = 'ping'
        resid = 'researcher_id_1234'
        ping_msg = {
            'command': command,
            'researcher_id': resid,
            'sequence': 1234
        }

        # action
        self.n1.on_message(ping_msg)

        # checks
        # check if `Messaging.send_message` has been called with good arguments
        msg_send_error_patch.assert_called_once_with(ErrorNumbers.FB301,
                                                     extra_msg='Message was not serializable',
                                                     researcher_id=resid)

    @patch('fedbiomed.node.round.Round.__init__')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_12_parser_task_train_create_round(self,
                                              node_msg_request_patch,
                                              history_monitor_patch,
                                              round_patch
                                              ):
        """Tests if rounds are created accordingly - running normal case scenario
        (in `parser_task_train` method)"""

        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        round_patch.return_value = None

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None
        # test 1: case where 1 dataset has been found
        msg_1_dataset = NodeMessages.request_create({
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training_plan_url': 'https://link.to.somewhere.where.my.model',
            'training_plan_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': 'researcher_id_1234',
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']}
        })

        # action
        self.n1.parser_task_train(msg_1_dataset)

        # checks
        # check that `Round` has been called once
        self.assertEqual(round_patch.call_count, 1)
        # check the attribute `rounds` of `Node` (should be a
        # list containing `Round` objects)
        self.assertEqual(len(self.n1.rounds), 1)
        self.assertIsInstance(self.n1.rounds[0], Round)
        # #####
        # test 2: case where 2 dataset have been found (training on several dataset)
        # reset mocks (for second test)
        round_patch.reset_mock()
        history_monitor_patch.reset_mock()
        round_patch.return_value = None

        # defining msg argument (case where 2 datasets are found)
        dict_msg_2_datasets = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training': True,
            'training_plan_url': 'https://link.to.somewhere.where.my.model',
            'training_plan_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': 'researcher_id_1234',
            'training_data': {environ['NODE_ID']: ['dataset_id_1234',
                                                   'dataset_id_6789']}
        }
        msg_2_datasets = NodeMessages.request_create(dict_msg_2_datasets)

        # action

        self.n2.parser_task_train(msg_2_datasets)

        # checks

        # FIXME: is this a good idea? Unit test may fail if
        # parameters are passed using arg name,
        # and if order change. Besides, it doesn't test
        # if value passed is a `HistoryMonitor` object (could be everything, test will pass)
        # see `sentinel` in unittests documentation
        # (difficult to use since we are patching constructor)

        round_patch.assert_called_with(dict_msg_2_datasets['model_args'],
                                       dict_msg_2_datasets['training_args'],
                                       True,
                                       self.database_id,
                                       dict_msg_2_datasets['training_plan_url'],
                                       dict_msg_2_datasets['training_plan_class'],
                                       dict_msg_2_datasets['params_url'],
                                       dict_msg_2_datasets['job_id'],
                                       dict_msg_2_datasets['researcher_id'],
                                       unittest.mock.ANY,  # this is for HistoryMonitor
                                       None,
                                       None,
                                       dlp_and_loading_block_metadata=None)

        # check if object `Round()` has been called twice
        self.assertEqual(round_patch.call_count, 2)
        self.assertEqual(len(self.n2.rounds), 2)
        # check if passed value is a `Round` object
        self.assertIsInstance(self.n2.rounds[0], Round)
        # check if object `HistoryMonitor` has been called
        history_monitor_patch.assert_called_once()
        # retrieve `HistoryMonitor` object
        history_monitor_ref = round_patch.call_args_list[-1][0][-3] 
        # `-3` cause HistoryMonitor object is the third last object passed in `Round` class

        # check id retrieve object is a HistoryMonitor object
        self.assertIsInstance(history_monitor_ref, HistoryMonitor)

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_13_parser_task_train_no_dataset_found(self,
                                                  node_msg_request_patch,
                                                  history_monitor_patch,
                                                  node_msg_reply_patch,
                                                  messaging_patch,
                                                  ):
        """Tests parser_task_train method, case where no dataset has been found """
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        history_monitor_patch.return_value = None

        # defining arguments
        resid = 'researcher_id_1234'
        msg_without_datasets = NodeMessages.request_create({
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training_plan_url': 'https://link.to.somewhere.where.my.model',
            'training_plan_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': resid,
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']}
        })
        # create tested object

        mock_dataset_manager = MagicMock()
        # return emtpy list to mimic dataset that havenot been found
        mock_dataset_manager.search_by_id = MagicMock(return_value=[])

        self.n1.dataset_manager = mock_dataset_manager

        # action

        self.n1.parser_task_train(msg_without_datasets)

        # checks
        messaging_patch.assert_called_once_with({
            'command': 'error',
            'node_id': environ['NODE_ID'],
            'researcher_id': resid,
            'errnum': ErrorNumbers.FB313,
            'extra_msg': "Did not found proper data in local datasets"
        })

    @patch('fedbiomed.node.round.Round.__init__')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    def test_node_14_parser_task_train_create_round_deserializer_str_msg(self,
                                                                   history_monitor_patch,
                                                                   round_patch
                                                                   ):
        """Tests if message is correctly deserialized if message is in string"""

        # defining arguments
        dict_msg_1_dataset = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training_plan_url': 'https://link.to.somewhere.where.my.model',
            'training_plan_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': 'researcher_id_1234',
            'command': 'train',
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']},
            'training': True,
            'aggregator_args': {}
        }
        # we convert this dataset into a string
        msg1_dataset = NodeMessages.request_create(dict_msg_1_dataset)

        # defining patchers
        round_patch.return_value = None
        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None

        # action
        self.n1.parser_task_train(msg1_dataset)

        # checks
        round_patch.assert_called_once_with(dict_msg_1_dataset['model_args'],
                                            dict_msg_1_dataset['training_args'],
                                            True,
                                            self.database_id,
                                            dict_msg_1_dataset['training_plan_url'],
                                            dict_msg_1_dataset['training_plan_class'],
                                            dict_msg_1_dataset['params_url'],
                                            dict_msg_1_dataset['job_id'],
                                            dict_msg_1_dataset['researcher_id'],
                                            unittest.mock.ANY,  # FIXME: should be an history monitor object
                                            None,
                                            None,
                                            dlp_and_loading_block_metadata=None
                                            )

    @patch('fedbiomed.node.round.Round.__init__')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    def test_node_15_parser_task_train_create_round_deserializer_bytes_msg(self,
                                                                     history_monitor_patch,
                                                                     round_patch
                                                                     ):
        """Tests if message is correctly deserialized if message is in bytes"""

        # defining arguments
        dict_msg_1_dataset = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training": True,
            "training_plan_url": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]},
            'aggregator_args': {}
        }

        #
        msg_1_dataset = NodeMessages.request_create(dict_msg_1_dataset)

        # defining patchers
        round_patch.return_value = None
        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None

        # action
        self.n1.parser_task_train(msg_1_dataset)

        # checks
        round_patch.assert_called_once_with(dict_msg_1_dataset['model_args'],
                                            dict_msg_1_dataset['training_args'],
                                            True,
                                            self.database_id,
                                            dict_msg_1_dataset['training_plan_url'],
                                            dict_msg_1_dataset['training_plan_class'],
                                            dict_msg_1_dataset['params_url'],
                                            dict_msg_1_dataset['job_id'],
                                            dict_msg_1_dataset['researcher_id'],
                                            unittest.mock.ANY,  # FIXME: should be an history_monitor object
                                            None, None,
                                            dlp_and_loading_block_metadata=None)

    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_16_parser_task_train_error_found(self,
                                             node_msg_request_patch,
                                             history_monitor_patch,
                                             ):
        """Tests correct raise of error (AssertionError) for missing/invalid
        entries in input arguments"""

        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        history_monitor_patch.return_value = None
        # FIXME: should we patch `validator` too (currently not patched) ?
        # validator_patch.return_value = True

        # test 1: test case where exception is raised when training_plan_url is None
        # defining arguments
        resid = 'researcher_id_1234'
        dict_msg_without_training_plan_url = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'training_plan_url': None,
            'training_plan_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': resid,
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']}
        }
        msg_without_training_plan_url = NodeMessages.request_create(dict_msg_without_training_plan_url)

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError`` is raised when `training_plan_url`entry is missing
            self.n1.parser_task_train(msg_without_training_plan_url)

        # test 2: test case where url is not valid
        dict_msg_with_unvalid_url = copy.deepcopy(dict_msg_without_training_plan_url)
        msg_with_unvalid_url = NodeMessages.request_create(dict_msg_with_unvalid_url)

        dict_msg_without_training_plan_url['training_plan_url'] = 'this is not a valid url'

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError` is raised when `training_plan_url` is invalid
            self.n1.parser_task_train(msg_with_unvalid_url)

        # test 3: test case where training_plan_class is None
        dict_msg_without_training_plan_class = copy.deepcopy(dict_msg_without_training_plan_url)
        dict_msg_without_training_plan_class['training_plan_class'] = None
        msg_without_training_plan_class = NodeMessages.request_create(dict_msg_without_training_plan_class)

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError` is raised when `training_plan_class` entry is not defined
            self.n1.parser_task_train(msg_without_training_plan_class)

        # test 4: test case where training_plan_class is not of type `str`
        dict_msg_training_plan_class_bad_type = copy.deepcopy(dict_msg_without_training_plan_url)
        # let's test with integer in place of strings
        dict_msg_training_plan_class_bad_type['training_plan_class'] = 1234
        msg_training_plan_class_bad_type = NodeMessages.request_create(dict_msg_training_plan_class_bad_type)

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError` is raised when `training_plan_class` entry is
            # of type string
            self.n1.parser_task_train(msg_training_plan_class_bad_type)

    def test_node_17_task_manager_normal_case_scenario(self):
        """Tests task_manager in the normal case scenario"""
        # TODO: implement such test when we will have methods that makes
        # possible to stop Node
        pass

    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_18_task_manager_exception_raised_task_queue(self,
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

    @patch('fedbiomed.node.node.Node.parser_task_train')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_19_task_manager_train_exception_raised_parser_task_train(self,
                                                               tasks_queue_get_patch,
                                                               node_parser_task_train_patch):
        """Tests case where `Node.parser_task_train` method raises an exception (SystemExit).
        """
        # defining patchers
        tasks_queue_get_patch.return_value = {}
        
        tasks_queue_get_patch.return_value = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "aggregator_args": {},
            "training": True,
            "training_plan_url": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]}
        }
        node_parser_task_train_patch.side_effect = SystemExit("mimicking an exception" + " coming from parser_task_train")  # noqa

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is caught
            # (should be triggered by `Node.parser_task_train` method)
            self.n1.task_manager()

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.NodeMessages.request_create')
    @patch('fedbiomed.node.node.NodeMessages.reply_create')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_20_task_manager_exception_raised(self,
                                                               tasks_queue_get_patch,
                                                               reply_create_patch,
                                                               request_create_patch,
                                                               mssging_send_msg_patch):
        """Tests case where `NodeMessages.request_create` method raises an exception 
        and then the reply_create raises another exception(SystemExit).
        """
        # defining patchers
        tasks_queue_get_patch.return_value = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training": True,
            "training_plan_url": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]}
        }
        request_create_patch.side_effect = Exception
        reply_create_patch.side_effect = SystemExit("mimicking an exception" + " coming from NodeMessages.request_create")  # noqa
        mssging_send_msg_patch.return_value = None

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is caught
            # (should be triggered by `Node.parser_task_train` method)
            self.n1.task_manager()

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node.parser_task_train')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_21_task_manager_train_exception_raised_send_message(self,
                                                                tasks_queue_get_patch,
                                                                node_parser_task_train_patch,
                                                                mssging_send_msg_patch):
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
            "training_plan_url": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]}
        }
        node_parser_task_train_patch.return_value = None
        mssging_send_msg_patch.side_effect = SystemExit("Mimicking an exception happening in" + "`send_message` method")  # noqa
        # defining arguments and attributes
        Round = MagicMock()
        Round.run_model_training = MagicMock(run_model_training=None)
        self.n1.rounds = [Round(), Round()]

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is caught (should be triggered by
            # `messaging.send_message` method )
            self.n1.task_manager()

    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node.parser_task_train')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_22_task_manager_train_exception_raised_task_done(self,
                                                             tasks_queue_get_patch,
                                                             node_parser_task_train_patch,
                                                             mssging_send_msg_patch,
                                                             tasks_queue_task_done_patch):
        """Tests if an Exception (SystemExit) is triggered when calling
        `TasksQueue.task_done` method for train message"""
        # defining patchers
        tasks_queue_get_patch.return_value = {}

        tasks_queue_get_patch.return_value = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "aggregator_args": {},
            "training": True,
            "training_plan_url": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]}
        }
        node_parser_task_train_patch.return_value = None
        mssging_send_msg_patch.return_value = None

        tasks_queue_task_done_patch.side_effect = SystemExit("Mimicking an exception happening in" + "`TasksQueue.task_done` method")  # noqa
        # defining arguments
        Round = MagicMock()
        Round.run_model_training = MagicMock(run_model_training=None)
        self.n1.rounds = [Round(), Round()]

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is raised (should be triggered by `TasksQueue.task_done`)
            self.n1.task_manager()

        # check that `Messaging.send_message` have been called twice
        # (because 2 rounds have been set in `rounds` attribute)
        self.assertEqual(mssging_send_msg_patch.call_count, 2)

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node._task_secagg')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_23_task_manager_secagg_exception_raised_task_done(
            self,
            tasks_queue_get_patch,
            task_secagg_patch,
            mssging_send_msg_patch,
            tasks_queue_task_done_patch,
            patch_servkey_manager,
            patch_biprime_manager):
        """Tests if an Exception (SystemExit) is triggered when calling
        `TasksQueue.task_done` method for secagg message"""
        # defining patchers
        tasks_queue_get_patch.return_value = {
            "researcher_id": "my_test_researcher",
            "secagg_id": "my_test_secagg",
            "sequence": 2345,
            "element": 33,
            "job_id": "my_job",
            "parties": [],
            "command": "secagg"
        }
        task_secagg_patch.return_value = None
        mssging_send_msg_patch.return_value = None

        tasks_queue_task_done_patch.side_effect = SystemExit("Mimicking an exception happening in" + "`TasksQueue.task_done` method")  # noqa

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is raised (should be triggered by `TasksQueue.task_done`)
            self.n1.task_manager()

        # check that `Messaging.send_message` has not been called
        self.assertEqual(mssging_send_msg_patch.call_count, 0)

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_24_task_manager_badcommand_exception_raised_task_done(
            self,
            tasks_queue_get_patch,
            mssging_send_msg_patch,
            tasks_queue_task_done_patch,
            patch_servkey_manager,
            patch_biprime_manager):
        """Tests if an Exception (SystemExit) is triggered when calling
        `TasksQueue.task_done` method for an unexpected type of message"""
        # defining patchers
        tasks_queue_get_patch.return_value = {
            "researcher_id": "researcher_id_1234",
            "secagg_id": "secagg_id_2345",
            "sequence": 33,
            "element": 1,
            "job_id": "my_test_job",
            "command": "secagg-delete",
        }
        mssging_send_msg_patch.return_value = None

        tasks_queue_task_done_patch.side_effect = SystemExit("Mimicking an exception happening in" + "`TasksQueue.task_done` method")  # noqa

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is raised (should be triggered by `TasksQueue.task_done`)
            self.n1.task_manager()

        # check that `Messaging.send_message` have been called once
        self.assertEqual(mssging_send_msg_patch.call_count, 1)

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node.parser_task_train')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_25_task_manager_train_exception_raised_twice_in_send_msg(self,
                                                                     tasks_queue_get_patch,
                                                                     node_parser_task_train_patch,
                                                                     mssging_send_msg_patch,
                                                                     tasks_queue_task_done_patch,
                                                                     node_msg_reply_create_patch):
        """
        Tests `task_manager` method, check what happens if `Messaging.send_message`
        triggers an exception.
        """
        # defining attributes

        Round = MagicMock()
        Round.run_model_training = MagicMock(run_model_training=None)
        self.n1.rounds = [Round()]  # only one item in the Round, so
        # second exception will be raised within the `except Exception as e` block
        # of `task_manager`

        # defining patchers
        tasks_queue_get_patch.return_value = {}

        tasks_queue_get_patch.return_value = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "aggregator_args": {},
            "training": True,
            "training_plan_url": "https://link.to.somewhere.where.my.model",
            "training_plan_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]}
        }
        node_parser_task_train_patch.return_value = None
        tasks_queue_task_done_patch.return_value = None
        node_msg_reply_create_patch.side_effect = TestNode.node_msg_side_effect
        mssging_send_msg_patch.side_effect = [Exception('mimicking exceptions'), SystemExit]

        # action
        with self.assertRaises(SystemExit):
            # checks if `task_manager` triggers SystemExit exception
            self.n1.task_manager()

        # checks if `Messaging.send_message` is called with
        # good arguments (second time it is called)
        mssging_send_msg_patch.assert_called_with(
            {
                'command': 'error',
                'extra_msg': str(Exception('mimicking exceptions')),
                'node_id': environ['NODE_ID'],
                'researcher_id': 'NOT_SET',
                'errnum': ErrorNumbers.FB300
            })

    @patch('fedbiomed.common.messaging.Messaging.start')
    def test_node_26_start_messaging_normal_case_scenario(self,
                                                          msg_start_patch):
        """Tests `start_messaging` method (normal_case_scenario)"""
        # arguments
        block = True

        # action
        self.n1.start_messaging(block)

        # checks
        msg_start_patch.assert_called_once_with(block)

    @patch('fedbiomed.common.messaging.Messaging.send_error')
    def test_node_27_send_error_normal_case_scenario(self, msg_send_error_patch):
        """Tests `send_error` method (normal case scenario)"""
        # arguments
        errnum = ErrorNumbers.FB100
        extra_msg = "this is a test_send_error"
        researcher_id = 'researcher_id_1224'

        # action
        self.n1.send_error(errnum, extra_msg, researcher_id)

        # checks
        msg_send_error_patch.assert_called_once_with(errnum,
                                                     extra_msg=extra_msg,
                                                     researcher_id=researcher_id)

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_28_on_message_search_privacy_obfuscation(self,
                                                           messaging_send_msg_patch
                                                           ):
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
            'command': 'search',
            'researcher_id': 'researcher_id_1234',
            'tags': ['#some_tags']
        }
        # action
        n3.on_message(search_msg)

        # check privacy-sensitive info a case-by-case basis
        database_info = messaging_send_msg_patch.call_args[0][0]['databases'][0]
        self.assertNotIn('path', database_info)
        self.assertNotIn('tabular_file', database_info['dataset_parameters'])

    @patch('fedbiomed.node.node.SecaggBiprimeSetup')
    @patch('fedbiomed.node.node.SecaggServkeySetup')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_29_task_secagg_success(
            self,
            messaging_send_msg_patch,
            secagg_servkey_patch,
            secagg_biprime_patch):
        """Tests `_task_secagg` normal (successful) case"""

        for el in [0, 1]:
            # prepare
            dict_secagg_request = {
                'researcher_id': 'my_test_researcher_id',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': el,
                'job_id': 'my_test_job',
                'parties': ['party1', 'party2', 'party3'],
                'command': 'secagg'
            }
            msg_secagg_request = NodeMessages.request_create(dict_secagg_request)
            dict_secagg_reply = {
                'researcher_id': dict_secagg_request['researcher_id'],
                'secagg_id': dict_secagg_request['secagg_id'],
                'sequence': dict_secagg_request['sequence'],
                'command': dict_secagg_request['command'],
                'node_id': environ['NODE_ID'],
                'success': True,
                'msg': ''
            }

            secagg_servkey_patch.return_value = FakeSecaggServkeySetup(
                dict_secagg_request['researcher_id'],
                dict_secagg_request['secagg_id'],
                dict_secagg_request['job_id'],
                dict_secagg_request['sequence'],
                dict_secagg_request['parties']
            )
            secagg_biprime_patch.return_value = FakeSecaggBiprimeSetup(
                dict_secagg_request['researcher_id'],
                dict_secagg_request['secagg_id'],
                dict_secagg_request['job_id'],
                dict_secagg_request['sequence'],
                dict_secagg_request['parties']
            )

            # action
            self.n1._task_secagg(msg_secagg_request)

            # check
            messaging_send_msg_patch.assert_called_with(dict_secagg_reply)

            self.assertEqual(secagg_servkey_patch.return_value.researcher_id(), dict_secagg_request['researcher_id'])
            self.assertEqual(secagg_servkey_patch.return_value.secagg_id(), dict_secagg_request['secagg_id'])
            self.assertEqual(secagg_servkey_patch.return_value.job_id(), dict_secagg_request['job_id'])
            self.assertEqual(secagg_servkey_patch.return_value.sequence(), dict_secagg_request['sequence'])
            self.assertEqual(secagg_biprime_patch.return_value.researcher_id(), dict_secagg_request['researcher_id'])
            self.assertEqual(secagg_biprime_patch.return_value.secagg_id(), dict_secagg_request['secagg_id'])
            self.assertEqual(secagg_biprime_patch.return_value.job_id(), dict_secagg_request['job_id'])
            self.assertEqual(secagg_biprime_patch.return_value.sequence(), dict_secagg_request['sequence'])

            messaging_send_msg_patch.reset_mock()

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_30_task_secagg_badmessage(
            self,
            messaging_send_msg_patch,
            patch_servkey_manager,
            patch_biprime_manager):
        """Tests `_task_secagg` with bad message values"""
        # this is not pure unit test as we don't mock SecaggServkeySetup SecaggBiprimeSetup

        # prepare
        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        bad_element = 2
        dict_secagg_requests = [
            {
                'researcher_id': 'party1',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': bad_element,
                'job_id': 'job1',
                'parties': ['party1', 'party2', 'party3'],
                'command': 'secagg'
            },
            {
                'researcher_id': '',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': 0,
                'job_id': 'job1',
                'parties': ['party1', 'party2', 'party3'],
                'command': 'secagg'
            },
            {
                'researcher_id': 'party1',
                'secagg_id': '',
                'sequence': 888,
                'element': 0,
                'job_id': 'job1',
                'parties': ['party1', 'party2', 'party3'],
                'command': 'secagg'
            },
            {
                'researcher_id': 'party1',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': 0,
                'job_id': 'job1',
                'parties': ['party1', 'party2'],
                'command': 'secagg'
            },
        ]
        dict_secagg_extra_msg = [
            f'ErrorNumbers.FB318: received bad request message: incorrect `element` {bad_element}',
            f'ErrorNumbers.FB318: bad secure aggregation request received by {environ["NODE_ID"]}: FB318: Secure '
            f'aggregation setup error: bad parameter `researcher_id` should not be empty string',
            f'ErrorNumbers.FB318: bad secure aggregation request received by {environ["NODE_ID"]}: FB318: Secure '
            f'aggregation setup error: bad parameter `secagg_id` should not be empty string',
            f"ErrorNumbers.FB318: bad secure aggregation request received by {environ['NODE_ID']}: FB318: Secure "
            f"aggregation setup error: bad parameter `parties` : ['party1', 'party2'] : need  at least 3 parties for secure aggregation",
        ]
        dict_secagg_reply_type = [
            "error",
            "reply",
            "reply",
            "reply"
        ]

        class CustomFakeMessages(FakeMessages):
            def get_param(self, val: str) -> Any:
                if val in self.msg:
                    return self.msg.get(val)
                else:
                    raise AttributeError(f"no such attribute '{val}'")

        for req in dict_secagg_requests:
            msg_secagg_request = CustomFakeMessages(req)

            reply_type = dict_secagg_reply_type.pop(0)

            if reply_type == 'error':
                dict_secagg_reply = {
                    'command': 'error',
                    'extra_msg': dict_secagg_extra_msg.pop(0),
                    'node_id': environ['NODE_ID'],
                    'researcher_id': req['researcher_id'],
                    'errnum': ErrorNumbers.FB318
                }
            else:
                dict_secagg_reply = {
                    'researcher_id': req['researcher_id'],
                    'secagg_id': req['secagg_id'],
                    'sequence': req['sequence'],
                    'success': False,
                    'node_id': environ['NODE_ID'],
                    'msg': dict_secagg_extra_msg.pop(0),
                    'command': 'secagg'
                }

            # action
            self.n1._task_secagg(msg_secagg_request)

            # check
            messaging_send_msg_patch.assert_called_with(dict_secagg_reply)
            # messaging_send_msg_patch.assert_not_called()
            messaging_send_msg_patch.reset_mock()

    @patch('fedbiomed.node.node.SecaggBiprimeSetup')
    @patch('fedbiomed.node.node.SecaggServkeySetup')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_31_task_secagg_fails_secagg_create(
            self,
            messaging_send_msg_patch,
            secagg_servkey_patch,
            secagg_biprime_patch):
        """Tests `_task_secagg` failing in secagg creation"""

        for el in [0, 1]:
            # prepare
            dict_secagg_request = {
                'researcher_id': 'my_test_researcher_id',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': el,
                'job_id': 'my_job',
                'parties': ['party1', 'party2', 'party3'],
                'command': 'secagg'
            }
            msg_secagg_request = NodeMessages.request_create(dict_secagg_request)
            dict_secagg_reply = {
                'researcher_id': dict_secagg_request['researcher_id'],
                'secagg_id': dict_secagg_request['secagg_id'],
                'sequence': dict_secagg_request['sequence'],
                'success': False,
                'node_id': environ['NODE_ID'],
                'msg': f"ErrorNumbers.FB318: bad secure aggregation request received by {environ['NODE_ID']}: ",
                'command': 'secagg'
            }

            secagg_servkey_patch.side_effect = Exception
            secagg_biprime_patch.side_effect = Exception

            # action
            self.n1._task_secagg(msg_secagg_request)

            # check
            messaging_send_msg_patch.assert_called_with(dict_secagg_reply)

            messaging_send_msg_patch.reset_mock()

    @patch('fedbiomed.node.node.SecaggBiprimeSetup')
    @patch('fedbiomed.node.node.SecaggServkeySetup')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_32_task_secagg_fails_secagg_setup(
            self,
            messaging_send_msg_patch,
            secagg_servkey_patch,
            secagg_biprime_patch):
        """Tests `_task_secagg` failing in `secagg.setup()`"""

        for el in [0, 1]:
            # prepare
            dict_secagg_request = {
                'researcher_id': 'my_test_researcher_id',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': el,
                'job_id': 'my_job',
                'parties': ['party1', 'party2', 'party3'],
                'command': 'secagg'
            }
            msg_secagg_request = NodeMessages.request_create(dict_secagg_request)
            dict_secagg_reply = {
                'researcher_id': dict_secagg_request['researcher_id'],
                'secagg_id': dict_secagg_request['secagg_id'],
                'sequence': dict_secagg_request['sequence'],
                'command': dict_secagg_request['command'],
                'node_id': environ['NODE_ID'],
                'success': False,
                'msg': f'ErrorNumbers.FB318: error during secagg setup for type {SecaggElementTypes(dict_secagg_request["element"])}: '
            }

            class FakeSecaggServkeySetupError(FakeSecaggServkeySetup):
                def setup(self):
                    raise Exception
            secagg_servkey_patch.return_value = FakeSecaggServkeySetupError(
                dict_secagg_request['researcher_id'],
                dict_secagg_request['secagg_id'],
                dict_secagg_request['job_id'],
                dict_secagg_request['sequence'],
                dict_secagg_request['parties']
            )

            class FakeSecaggBiprimeSetupError(FakeSecaggBiprimeSetup):
                def setup(self):
                    raise Exception
            secagg_biprime_patch.return_value = FakeSecaggBiprimeSetupError(
                dict_secagg_request['researcher_id'],
                dict_secagg_request['secagg_id'],
                dict_secagg_request['job_id'],
                dict_secagg_request['sequence'],
                dict_secagg_request['parties']
            )

            # action
            self.n1._task_secagg(msg_secagg_request)

            # check
            messaging_send_msg_patch.assert_called_with(dict_secagg_reply)
            messaging_send_msg_patch.reset_mock()

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.node.node.SecaggElementTypes')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_33_task_secagg_fails_secagg_bad_secagg_element(
            self,
            messaging_send_msg_patch,
            element_types_patch,
            patch_servkey_manager,
            patch_biprime_manager):
        """Tests `_task_secagg` and `_task_secgg_delete` failing with bad secagg element type"""

        # prepare
        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        bad_message_values = [2, 18, 987]
        test_setups = [
            [
                'secagg',
                f'ErrorNumbers.FB318: bad secure aggregation request message received by {environ["NODE_ID"]}: ',
                self.n1._task_secagg
            ],
            [
                'secagg-delete',
                f'ErrorNumbers.FB321: bad secagg delete request message received by {environ["NODE_ID"]}: ',
                self.n1._task_secagg_delete
            ],
        ]
        for bad_message_value in bad_message_values:

            for command, message, function in test_setups:
                class FakeSecaggElementTypes(_BaseEnum):
                    DUMMY: int = bad_message_value
                element_types_patch.return_value = FakeSecaggElementTypes(bad_message_value)
                element_types_patch.__iter__.return_value = [
                    FakeSecaggElementTypes(bad_message_value)
                ]

                dict_secagg_request = {
                    'researcher_id': 'my_test_researcher_id',
                    'secagg_id': 'my_dummy_secagg_id',
                    'sequence': 888,
                    'element': bad_message_value,
                    'job_id': 'my_job_id',
                    'command': command
                }
                if command == 'secagg':
                    dict_secagg_request['parties'] = ['party1', 'party2', 'party3']
                msg_secagg_request = NodeMessages.request_create(dict_secagg_request)
                dict_secagg_reply = {
                    'researcher_id': dict_secagg_request['researcher_id'],
                    'secagg_id': dict_secagg_request['secagg_id'],
                    'sequence': dict_secagg_request['sequence'],
                    'command': dict_secagg_request['command'],
                    'node_id': environ['NODE_ID'],
                    'success': False,
                    'msg': f'{message}no such element {FakeSecaggElementTypes(dict_secagg_request["element"]).name}'
                }

                # action
                function(msg_secagg_request)

                # check
                messaging_send_msg_patch.assert_called_with(dict_secagg_reply)
                messaging_send_msg_patch.reset_mock()

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_node_34_task_secagg_delete_badmessage(
            self,
            messaging_send_msg_patch,
            patch_servkey_manager,
            patch_biprime_manager):
        """Tests `_task_secagg_delete` with bad message values"""
        # this is not pure unit test as we don't mock SecaggServkeySetup SecaggBiprimeSetup

        # prepare
        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        bad_element = 2
        dict_secagg_delete_requests = [
            {
                'researcher_id': 'party1',
                'secagg_id': 'my_dummy_secagg_id',
                'sequence': 888,
                'element': bad_element,
                'job_id': 'job1',
                'command': 'secagg-delete'
            },
            # no such element in database
            {
                'researcher_id': '',
                'secagg_id': 'my_dummy_secagg_id_NOT_IN_DB',
                'sequence': 888,
                'element': 0,
                'job_id': 'job1',
                'command': 'secagg-delete'
            },
        ]
        dict_secagg_delete_extra_msg = [
            f'ErrorNumbers.FB321: received bad delete message: incorrect `element` {bad_element}',
            f'ErrorNumbers.FB321: no such secagg context element in node database for node_id={environ["NODE_ID"]} secagg_id=my_dummy_secagg_id_NOT_IN_DB',
        ]
        dict_secagg_delete_reply_type = [
            "error",
            "reply",
        ]

        class CustomFakeMessages(FakeMessages):
            def get_param(self, val: str) -> Any:
                if val in self.msg:
                    return self.msg.get(val)
                else:
                    raise AttributeError(f"no such attribute '{val}'")

        for req in dict_secagg_delete_requests:
            msg_secagg_delete_request = CustomFakeMessages(req)

            reply_type = dict_secagg_delete_reply_type.pop(0)

            if reply_type == 'error':
                dict_secagg_delete_reply = {
                    'command': 'error',
                    'extra_msg': dict_secagg_delete_extra_msg.pop(0),
                    'node_id': environ['NODE_ID'],
                    'researcher_id': req['researcher_id'],
                    'errnum': ErrorNumbers.FB321
                }
            else:
                dict_secagg_delete_reply = {
                    'researcher_id': req['researcher_id'],
                    'secagg_id': req['secagg_id'],
                    'sequence': req['sequence'],
                    'success': False,
                    'node_id': environ['NODE_ID'],
                    'msg': dict_secagg_delete_extra_msg.pop(0),
                    'command': 'secagg-delete'
                }

            # action
            self.n1._task_secagg_delete(msg_secagg_delete_request)

            # check
            messaging_send_msg_patch.assert_called_with(dict_secagg_delete_reply)
            # messaging_send_msg_patch.assert_not_called()
            messaging_send_msg_patch.reset_mock()

    @patch('fedbiomed.node.secagg.SecaggBiprimeManager')
    @patch('fedbiomed.node.secagg.SecaggServkeyManager')
    @patch('fedbiomed.node.node.SecaggBiprimeManager.remove')
    @patch('fedbiomed.node.node.SecaggServkeyManager.remove')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_35_task_secagg_delete_remove_error(
            self,
            node_msg_request_patch,
            node_msg_reply_patch,
            messaging_send_msg_patch,
            secagg_servkey_manager_remove_patch,
            secagg_biprime_manager_remove_patch,
            patch_servkey_manager,
            patch_biprime_manager
    ):
        """Tests secagg-delete command with error in `remove` command"""
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        secagg_servkey_manager_remove_patch.side_effect = Exception
        secagg_biprime_manager_remove_patch.side_effect = Exception

        patch_servkey_manager.return_value = FakeSecaggServkeyManager()
        patch_biprime_manager.return_value = FakeSecaggBiprimeManager()

        # defining arguments
        secagg_delete = {
            'command': 'secagg-delete',
            'researcher_id': 'researcher_id_1234',
            'secagg_id': 'my_test_secagg_id',
            'sequence': 1234,
            'element': 0,
            'job_id': 'a_dummy_job_id',
        }
        secagg_delete_reply = {
            'command': 'secagg-delete',
            'researcher_id': 'researcher_id_1234',
            'secagg_id': 'my_test_secagg_id',
            'sequence': 1234,
            'success': False,
            'node_id': environ['NODE_ID'],
            'msg': f'ErrorNumbers.FB321: error during secagg delete on node_id={environ["NODE_ID"]} secagg_id={secagg_delete["secagg_id"]}: ',
        }

        class CustomFakeMessages(FakeMessages):
            def get_param(self, val: str) -> Any:
                if val in self.msg:
                    return self.msg.get(val)
                else:
                    raise AttributeError(f"no such attribute '{val}'")


        # action
        self.n1._task_secagg_delete(CustomFakeMessages(secagg_delete))

        # checks
        messaging_send_msg_patch.assert_called_once_with(secagg_delete_reply)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
