# Managing NODE, RESEARCHER environ mock before running tests
from platform import node
from typing import Any, Dict
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

from fedbiomed.node.environ import environ

import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.node.node import Node


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

    @patch('fedbiomed.common.messaging.Messaging.__init__')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.__init__')
    def setUp(self,
              task_queue_patcher, 
              messaging_patcher):
            
        task_queue_patcher.return_value = None
        messaging_patcher.return_value = None
        mock_data_manager = MagicMock()
        mock_data_manager.search_by_tags = MagicMock(return_value=[{'database_id': '1234', 'path': '/path/to/my/dataset'}])
        mock_model_manager = MagicMock()
        
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
        self.n1.on_message(train_msg)
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
        # definiing arguments
        ping_msg = {
                'command': 'ping',
                'researcher_id': 'researcher_id_1234',
                'sequence': 1234
            }
        self.n1.on_message(ping_msg)
        ping_msg.update(
                        {
                            'node_id': environ['NODE_ID'],
                            'command': 'pong',
                            'success': True
                            })
        messaging_send_msg_patch.assert_called_once_with(ping_msg)
    
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.message.NodeMessages.request_create')
    def test_on_message_03_normal_case_scenario_search(self,
                                                       node_msg_request_patch,
                                                       node_msg_reply_patch,
                                                       messaging_send_msg_patch
                                                       ):
        
        def side_effect(*args, **kwargs):
            print("OK")
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
    
        node_msg_reply_patch.side_effect = TestNode.node_msg_side_effect
        database_val = [{'database_id': '1234', 'path': '/path/to/my/dataset'}]
        # defining arguments
        search_msg = {
            'command': 'search',
            'researcher_id': 'researcher_id_1234',
            'tags': ['#some_tags']
        }
        self.n1.on_message(search_msg)
        
        # argument `search_msg` will be modified: we will check if 
        # message has been modified accordingly
        database_val[0].pop('path', None)
        search_msg.pop('tags', None)
        search_msg.update({'success': True,
                           'node_id': environ['NODE_ID'],
                           'databases': database_val,
                           'count': len(database_val)})
        messaging_send_msg_patch.assert_called_once_with(search_msg)
