# Managing NODE, RESEARCHER environ mock before running tests
from platform import node
from typing import Any, Dict
from fedbiomed.common.constants import ErrorNumbers
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

from fedbiomed.node.environ import environ

import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.node.node import Node

from json import decoder


class TestNode(unittest.TestCase):
    
    # Fake classes definition
    class FakeNodeMessages:
        # Fake NodeMessage class
        def __init__(self, msg: Dict[str, Any]):
            self.msg = msg

        def get_dict(self) -> Dict[str, Any]:
            return self.msg
    
    @classmethod
    def setUpClass(cls):
        # defining common side effect functions
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = TestNode.FakeNodeMessages(msg)
            return fake_node_msg
        
        cls.node_msg_side_effect = node_msg_side_effect

    @patch('fedbiomed.common.messaging.Messaging.__init__', autospec=True)
    @patch('fedbiomed.common.tasks_queue.TasksQueue.__init__', autospec=True)
    def setUp(self,
              task_queue_patcher,
              messaging_patcher):

        
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
        
        # patchers
        task_queue_patcher.return_value = None
        messaging_patcher.return_value = None
        
        # mocks
        mock_data_manager = MagicMock()
        mock_data_manager.search_by_tags = MagicMock(return_value=self.database_val)
        mock_data_manager.list_my_data = MagicMock(return_value = self.database_list)
        mock_model_manager = MagicMock()
        mock_data_manager.reply_model_status_request = MagicMock(return_value = None)
        self.model_manager_mock = mock_model_manager
        
        self.n1 = Node(mock_data_manager, mock_model_manager)

    
    def tearDown(self) -> None:
        pass
    
    @patch('fedbiomed.common.tasks_queue.TasksQueue.add')
    def test_add_task(self, task_queue_add_patcher):
        """Tests add_task method"""
        # a dummy message
        node_msg_request_create_task = {
            'msg': "a message for testing",
            'command': 'train'
        }
        self.n1.add_task(node_msg_request_create_task)
        task_queue_add_patcher.assert_called_once_with(node_msg_request_create_task)
        
    @patch('fedbiomed.node.node.Node.add_task')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_on_message_01_normal_case_scenario_train_reply(
                                           self,
                                           node_msg_req_create_patcher,
                                           node_add_task_patcher,
                                           
                                           ):
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
    def test_on_message_02_normal_case_scenario_ping(
                                self,
                                node_msg_request_patch,
                                node_msg_reply_patch,
                                messaging_send_msg_patch
                                ):
        
        
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
    def test_on_message_03_normal_case_scenario_search(self,
                                                       node_msg_request_patch,
                                                       node_msg_reply_patch,
                                                       messaging_send_msg_patch
                                                       ):
        
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
    def test_on_message_04_normal_case_scenario_list(self,
                                                     node_msg_request_patch,
                                                     node_msg_reply_patch,
                                                     messaging_send_msg_patch
                                                     ):
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
        # Messaging class
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
    def test_on_message_05_normal_case_scenario_model_status(self,
                                                             node_msg_request_patch,
                                                             ):
        """Tests normal case senario, if command is equals to 'model-status"""
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
    def test_on_message_06_unknown_command(self,
                                           node_msg_request_patch,
                                           send_err_patch):
        """Tests Exception is raised if command is not a knwon command"""
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
    def test_on_message_07_fail_reading_json(self,
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
                                                     extra_msg = "Not able to deserialize the message",
                                                     researcher_id= resid)
     
    @patch('fedbiomed.node.node.Node.send_error')    
    def test_on_message_08_fail_getting_msg_field(self,
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
                                                     extra_msg = "'command' property was not found",
                                                     researcher_id= resid)
    
    @patch('fedbiomed.node.node.Node.send_error') 
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')  
    def test_on_message_09_fail_msg_not_serializable(self,
                                                     node_msg_request_patch,
                                                     node_msg_reply_patch,
                                                     messaging_patch,
                                                     msg_send_error_patch):
        """Tests case where a TypError is raised (because unable to serialize message)"""
        
        def messaging_side_effect(*args, **kwargs):
            raise TypeError('Mimicking a TypeError happening when serializing message')
        
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
                                                     extra_msg = 'Message was not serializable',
                                                     researcher_id= resid)   
