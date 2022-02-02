# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.node.node import Node

class TestNode(unittest.TestCase):
    
    @patch('fedbiomed.common.messaging.Messaging.__init__')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.__init__')
    def setUp(self,
              task_queue_patcher, 
              messaging_patcher):
        
        task_queue_patcher.return_value = None
        messaging_patcher.return_value = None
        mock_data_manager = MagicMock()
        mock_model_manager = MagicMock()
        
        self.n1 = Node(mock_data_manager, mock_model_manager)
    
    def tearDown(self) -> None:
        pass
    
    @patch('fedbiomed.common.tasks_queue.TasksQueue.add')
    def test_add_task(self, task_queue_add_patcher):
        
        # a dummy message
        node_msg_request_create = {
            'msg': "a message for testing",
            'command': 'train'
        }
        self.n1.add_task(node_msg_request_create)
        task_queue_add_patcher.assert_called_once_with(node_msg_request_create)
        
    