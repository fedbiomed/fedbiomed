import copy
import json
from json import decoder
import json
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch

import testsupport.mock_node_environ  # noqa (remove flake8 false warning)

# import dummy classes
from testsupport.fake_message import FakeMessages

from fedbiomed.node.environ import environ
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.node import Node
from fedbiomed.node.round import Round



class TestNode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        mock_dataset_manager = MagicMock()
        mock_dataset_manager.search_by_tags = MagicMock(return_value=self.database_val)
        mock_dataset_manager.list_my_data = MagicMock(return_value=self.database_list)
        mock_model_manager = MagicMock()
        mock_dataset_manager.reply_model_status_request = MagicMock(return_value=None)
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
        # arguments
        # a dummy message
        node_msg_request_create_task = {
            'msg': "a message for testing",
            'command': 'train'
        }
        # action
        self.n1.add_task(node_msg_request_create_task)

        # checks
        task_queue_add_patcher.assert_called_once_with(node_msg_request_create_task)

    @patch('fedbiomed.node.node.Node.add_task')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_02_on_message_normal_case_scenario_train_reply(
            self,
            node_msg_req_create_patcher,
            node_add_task_patcher,
    ):
        """Tests `on_message` method (normal case scenario), with train command"""
        # test 1: test normal case scenario, where `command` = 'train'

        node_msg_req_create_patcher.side_effect = TestNode.node_msg_side_effect
        train_msg = {
            'command': 'train'
        }
        # action
        self.n1.on_message(train_msg)

        # checks
        node_add_task_patcher.assert_called_once_with(train_msg)

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

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_04_on_message_normal_case_scenario_search(self,
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
    def test_node_05_on_message_normal_case_scenario_list(self,
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
    def test_node_06_on_message_normal_case_scenario_model_status(self,
                                                                  node_msg_request_patch,
                                                                  ):
        """Tests normal case scenario, if command is equals to 'model-status"""
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        # defining arguments
        model_status_msg = {
            'command': 'model-status',
            'researcher_id': 'researcher_id_1234',
        }

        # action
        self.n1.on_message(model_status_msg)

        # checks
        self.model_manager_mock.reply_model_status_request.assert_called_once_with(model_status_msg,
                                                                                   self.n1.messaging)

    @patch('fedbiomed.node.node.Node.send_error')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_07_on_message_unknown_command(self,
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
    def test_node_08_on_message_fail_reading_json(self,
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
    def test_node_09_on_message_fail_getting_msg_field(self,
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
    def test_node_10_on_message_fail_msg_not_serializable(self,
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
    def test_node_11_parser_task_create_round(self,
                                              node_msg_request_patch,
                                              history_monitor_patch,
                                              round_patch
                                              ):
        """Tests if rounds are created accordingly - running normal case scenario
        (in `parser_task` method)"""

        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        round_patch.return_value = None

        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None
        # test 1: case where 1 dataset has been found
        dict_msg_1_dataset = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'model_url': 'https://link.to.somewhere.where.my.model',
            'model_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': 'researcher_id_1234',
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']}
        }

        # action
        self.n1.parser_task(dict_msg_1_dataset)

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
            'model_url': 'https://link.to.somewhere.where.my.model',
            'model_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': 'researcher_id_1234',
            'training_data': {environ['NODE_ID']: ['dataset_id_1234',
                                                   'dataset_id_6789']}
        }

        # action

        self.n2.parser_task(dict_msg_2_datasets)

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
                                       dict_msg_2_datasets['model_url'],
                                       dict_msg_2_datasets['model_class'],
                                       dict_msg_2_datasets['params_url'],
                                       dict_msg_2_datasets['job_id'],
                                       dict_msg_2_datasets['researcher_id'],
                                       unittest.mock.ANY,
                                       None,
                                       dlp_metadata=None)

        # check if object `Round()` has been called twice
        self.assertEqual(round_patch.call_count, 2)
        self.assertEqual(len(self.n2.rounds), 2)
        # check if passed value is a `Round` object
        self.assertIsInstance(self.n2.rounds[0], Round)
        # check if object `HistoryMonitor` has been called
        history_monitor_patch.assert_called_once()
        # retrieve `HistoryMonitor` object
        history_monitor_ref = round_patch.call_args_list[-1][0][-2]
        # check id retrieve object is a HistoryMonitor object
        self.assertIsInstance(history_monitor_ref, HistoryMonitor)

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_12_parser_task_no_dataset_found(self,
                                                  node_msg_request_patch,
                                                  history_monitor_patch,
                                                  node_msg_reply_patch,
                                                  messaging_patch,
                                                  ):
        """Tests parser_task method, case where no dataset has been found """
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        history_monitor_patch.return_value = None

        # defining arguments
        resid = 'researcher_id_1234'
        dict_msg_without_datasets = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'model_url': 'https://link.to.somewhere.where.my.model',
            'model_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': resid,
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']}
        }
        # create tested object

        mock_dataset_manager = MagicMock()
        # return emtpy list to mimic dataset that havenot been found
        mock_dataset_manager.search_by_id = MagicMock(return_value=[])

        self.n1.dataset_manager = mock_dataset_manager

        # action

        self.n1.parser_task(dict_msg_without_datasets)

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
    def test_node_13_parser_task_create_round_deserializer_str_msg(self,
                                                                   history_monitor_patch,
                                                                   round_patch
                                                                   ):
        """Tests if message is correctly deserialized if message is in string"""

        # defining arguments
        dict_msg_1_dataset = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'model_url': 'https://link.to.somewhere.where.my.model',
            'model_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': 'researcher_id_1234',
            'command': 'train',
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']},
            'training': True
        }
        # we convert this dataset into a string
        incoming_msg = json.dumps(dict_msg_1_dataset)

        # defining patchers
        round_patch.return_value = None
        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None

        # action
        self.n1.parser_task(incoming_msg)

        # checks
        round_patch.assert_called_once_with(dict_msg_1_dataset['model_args'],
                                            dict_msg_1_dataset['training_args'],
                                            True,
                                            self.database_id,
                                            dict_msg_1_dataset['model_url'],
                                            dict_msg_1_dataset['model_class'],
                                            dict_msg_1_dataset['params_url'],
                                            dict_msg_1_dataset['job_id'],
                                            dict_msg_1_dataset['researcher_id'],
                                            unittest.mock.ANY,  # FIXME: should be an history monitor object
                                            None,
                                            dlp_metadata=None
                                            )

    @patch('fedbiomed.node.round.Round.__init__')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    def test_node_14_parser_task_create_round_deserializer_bytes_msg(self,
                                                                     history_monitor_patch,
                                                                     round_patch
                                                                     ):
        """Tests if message is correctly deserialized if message is in bytes"""

        # defining arguments
        dict_msg_1_dataset = {
            "model_args": {"lr": 0.1},
            "training_args": {"some_value": 1234},
            "training": True,
            "model_url": "https://link.to.somewhere.where.my.model",
            "model_class": "my_test_training_plan",
            "params_url": "https://link.to_somewhere.where.my.model.parameters.is",
            "job_id": "job_id_1234",
            "researcher_id": "researcher_id_1234",
            "command": "train",
            "training_data": {environ["NODE_ID"]: ["dataset_id_1234"]}
        }

        #
        incoming_msg = bytes( json.dumps(dict_msg_1_dataset) , 'utf-8')

        # defining patchers
        round_patch.return_value = None
        history_monitor_patch.spec = True
        history_monitor_patch.return_value = None

        # action
        self.n1.parser_task(incoming_msg)

        # checks
        round_patch.assert_called_once_with(dict_msg_1_dataset['model_args'],
                                            dict_msg_1_dataset['training_args'],
                                            True,
                                            self.database_id,
                                            dict_msg_1_dataset['model_url'],
                                            dict_msg_1_dataset['model_class'],
                                            dict_msg_1_dataset['params_url'],
                                            dict_msg_1_dataset['job_id'],
                                            dict_msg_1_dataset['researcher_id'],
                                            unittest.mock.ANY,  # FIXME: should be an history_monitor object
                                            None,
                                            dlp_metadata=None)

    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_node_15_parser_task_error_found(self,
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

        # test 1: test case where exception is raised when model_url is None
        # defining arguments
        resid = 'researcher_id_1234'
        dict_msg_without_model_url = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
            'model_url': None,
            'model_class': 'my_test_training_plan',
            'params_url': 'https://link.to_somewhere.where.my.model.parameters.is',
            'job_id': 'job_id_1234',
            'researcher_id': resid,
            'training_data': {environ['NODE_ID']: ['dataset_id_1234']}
        }

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError`` is raised when `model_url`entry is missing
            self.n1.parser_task(dict_msg_without_model_url)

        # test 2: test case where url is not valid
        dict_msg_with_unvalid_url = copy.deepcopy(dict_msg_without_model_url)
        dict_msg_without_model_url['model_url'] = 'this is not a valid url'

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError` is raised when `model_url` is invalid
            self.n1.parser_task(dict_msg_with_unvalid_url)

        # test 3: test case where model_class is None
        dict_msg_without_model_class = copy.deepcopy(dict_msg_without_model_url)
        dict_msg_without_model_class['model_class'] = None

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError` is raised when `model_class` entry is not defined
            self.n1.parser_task(dict_msg_without_model_class)

        # test 4: test case where model_class is not of type `str`
        dict_msg_model_class_bad_type = copy.deepcopy(dict_msg_without_model_url)
        # let's test with integer in place of strings
        dict_msg_model_class_bad_type['model_class'] = 1234

        # action
        with self.assertRaises(AssertionError):
            # checks if `AssertionError` is raised when `model_class` entry is
            # of type string
            self.n1.parser_task(dict_msg_model_class_bad_type)

    def test_node_16_task_manager_normal_case_scenario(self):
        """Tests task_manager in the normal case scenario"""
        # TODO: implement such test when we will have methods that makes
        # possible to stop Node
        pass

    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_17_task_manager_exception_raised_task_queue(self,
                                                              tasks_queue_get_patch):
        """Simulates an Exception (SystemError) triggered by `tasks_queue.get`
        """
        # defining patchers
        tasks_queue_get_patch.side_effect = SystemError("mimicking an exception coming from ")

        # action
        with self.assertRaises(SystemError):
            # checks if `SystemError` is caught (triggered by patched `tasks_queue.get`)
            self.n1.task_manager()

    # NOTA BENE: for test 14 to test 17 (testing `task_manager` method)
    # Since we don't have any proper way to stop the infinite loop defined
    # in the method, we are triggering `SystemExit` Exception to leave it
    # (SystemExit is an exception that is not caught by statement
    # `except Exception as e:`). When a more graceful way of exiting infinite loop
    # will be created, those tests should be updated

    @patch('fedbiomed.node.node.Node.parser_task')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_18_task_manager_exception_raised_parser_task(self,
                                                               tasks_queue_get_patch,
                                                               node_parser_task_patch):
        """Tests case where `tasks_queue.get` method raises an exception (SystemExit).
        """
        # defining patchers
        tasks_queue_get_patch.return_value = None
        node_parser_task_patch.side_effect = SystemExit("mimicking an exception" + " coming from parser_task")  # noqa

        # action
        with self.assertRaises(SystemExit):
            # checks if `SystemExit` is caught
            # (should be triggered by `tasks_queue.get` method)
            self.n1.task_manager()

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node.parser_task')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_19_task_manager_exception_raised_send_message(self,
                                                                tasks_queue_get_patch,
                                                                node_parser_task_patch,
                                                                mssging_send_msg_patch):
        """Tests case where `messaging.send_message` method
        raises an exception (SystemExit).
        """
        # defining patchers
        tasks_queue_get_patch.return_value = None
        node_parser_task_patch.return_value = None
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
    @patch('fedbiomed.node.node.Node.parser_task')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_20_task_manager_exception_raised_task_done(self,
                                                             tasks_queue_get_patch,
                                                             node_parser_task_patch,
                                                             mssging_send_msg_patch,
                                                             tasks_queue_task_done_patch):
        """Tests if an Exception (SystemExit) is triggered when calling
        `TasksQueue.task_done` method"""
        # defining patchers
        tasks_queue_get_patch.return_value = None
        node_parser_task_patch.return_value = None
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

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node.parser_task')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_node_21_task_manager_exception_raised_twice_in_send_msg(self,
                                                                     tasks_queue_get_patch,
                                                                     node_parser_task_patch,
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
        tasks_queue_get_patch.return_value = None
        node_parser_task_patch.return_value = None
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
    def test_node_22_start_messaging_normal_case_scenario(self,
                                                          msg_start_patch):
        """Tests `start_messaging` method (normal_case_scenario)"""
        # arguments
        block = True

        # action
        self.n1.start_messaging(block)

        # checks
        msg_start_patch.assert_called_once_with(block)

    @patch('fedbiomed.common.messaging.Messaging.send_error')
    def test_node_23_send_error_normal_case_scenario(self, msg_send_error_patch):
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
