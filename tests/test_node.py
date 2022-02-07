# Managing NODE, RESEARCHER environ mock before running tests
from asyncio import threads
from copy import deepcopy
import copy
import multiprocessing
from platform import node
import threading
import time
from typing import Any, Dict
from unittest import mock

from numpy import round_
from sklearn.linear_model import HuberRegressor
from fedbiomed.common import tasks_queue
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.node.history_monitor import HistoryMonitor
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

from fedbiomed.node.environ import environ

import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.node.node import Node
from fedbiomed.node.round import Round

from json import decoder


class TestNode(unittest.TestCase):
    
    # Fake classes definition
    class FakeNodeMessages:
        # Fake NodeMessage class
        def __init__(self, msg: Dict[str, Any]):
            self.msg = msg

        def get_dict(self) -> Dict[str, Any]:
            return self.msg
        
        def get_param(self, val: str) -> Any:
            return self.msg.get(val)
    
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
        
        self.database_id = [
            {
                'database_id': '1234',
                'path': '/path/to/my/dataset',
                'name': 'test_dataset1'
            }
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
        mock_data_manager.search_by_id = MagicMock(return_value = self.database_id)
        
        self.model_manager_mock = mock_model_manager
        
        self.n1 = Node(mock_data_manager, mock_model_manager)
        self.n2 = Node(mock_data_manager, mock_model_manager)
        
    
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
        # JSONDecodeError can be raised from messaging class
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
        
        # check
        msg_send_error_patch.assert_called_once_with(ErrorNumbers.FB301,
                                                     extra_msg = 'Message was not serializable',
                                                     researcher_id= resid)   
    
    @patch('fedbiomed.node.round.Round.__init__')
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__', spec=True)
    @patch('fedbiomed.common.message.NodeMessages.request_create') 
    def test_parser_task_01_create_round(self,
                                         node_msg_request_patch,
                                         history_monitor_patch,
                                         round_patch
                                         ):
        """Tests if rounds are created accordingly"""

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
        self.assertEqual(round_patch.call_count, 1)
        self.assertEqual(len(self.n1.rounds), 1)
        self.assertIsInstance(self.n1.rounds[0], Round)
        
        
        # test 2: case where 2 dataset have been found (training on several dataset)
        # reset mocks
        round_patch.reset_mock()
        round_patch.return_value = None
        
        # defining msg argument (where 2 datasets are found)
        dict_msg_2_datasets = {
            'model_args': {'lr': 0.1},
            'training_args': {'some_value': 1234},
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
        
        # hack to get the object HistoryMonitor
        # FIXME: is this a good idea? Unit test may fail if 
        # parameters are passed using arg name, 
        # and if order change. Besides, it doesnot test
        # if value passed is a `Round` object (could be everything, test will pass)
        # see `sentinel` in unitests documentation 
        # (difficult to use since we are patching constructor)
        
        round_patch.assert_called_with(dict_msg_2_datasets['model_args'],
                                        dict_msg_2_datasets['training_args'],
                                        self.database_id[0],
                                        dict_msg_2_datasets['model_url'],
                                        dict_msg_2_datasets['model_class'],
                                        dict_msg_2_datasets['params_url'],
                                        dict_msg_2_datasets['job_id'],
                                        dict_msg_2_datasets['researcher_id'],
                                        mock.ANY,
                                        None
                                    )
        self.assertEqual(round_patch.call_count, 2)
        self.assertEqual(len(self.n2.rounds), 2)
        # check if passed value is a `Round` object
        self.assertIsInstance(self.n2.rounds[0], Round)
        
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.common.message.NodeMessages.reply_create') 
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.request_create') 
    def test_parser_task_02_no_dataset_found(self,
                                             node_msg_request_patch,
                                             history_monitor_patch,
                                             node_msg_reply_patch,
                                             messaging_patch,
                                             ):
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
        
        mock_data_manager = MagicMock()
        # return emtpy list to mimic dataset that havenot been found
        mock_data_manager.search_by_id = MagicMock(return_value = [])  
        
        self.n1.data_manager = mock_data_manager
        
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
        
    @patch('fedbiomed.node.history_monitor.HistoryMonitor.__init__')
    @patch('fedbiomed.common.message.NodeMessages.request_create') 
    def test_parser_task_03_error_found(self,
                                        node_msg_request_patch,
                                        history_monitor_patch,
                                        ):
        """Tests correct raise of error (AssertionError) for missing/invalid 
        entries in input arguments"""
        
        # defining patchers
        node_msg_request_patch.side_effect = TestNode.node_msg_side_effect
        history_monitor_patch.return_value = None
        #validator_patch.return_value = True
        
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
            self.n1.parser_task(dict_msg_without_model_url)
            
        # test 2: test case where url is not valid
        dict_msg_with_unvalid_url = copy.deepcopy(dict_msg_without_model_url)
        dict_msg_without_model_url['model_url'] =  'this is not a valid url'
        
        #validator_patch.return_value = False
        
        # action
        with self.assertRaises(AssertionError):
            self.n1.parser_task(dict_msg_with_unvalid_url)
            
        # test 3: test case where model_class is None
        dict_msg_without_model_class = copy.deepcopy(dict_msg_without_model_url)
        dict_msg_without_model_class['model_class'] = None
        #validator_patch.return_value = True
        
         # action
        with self.assertRaises(AssertionError):
            self.n1.parser_task(dict_msg_without_model_class)
            
        # test 4: test case where model_class is not of type `str`
        dict_msg_model_class_bad_type = copy.deepcopy(dict_msg_without_model_url)
        # lets test with integer in place of strings
        dict_msg_model_class_bad_type['model_class'] = 1234  
        
        # action
        with self.assertRaises(AssertionError):
            self.n1.parser_task(dict_msg_model_class_bad_type)
    
    @patch('fedbiomed.node.round.Round.__init__')        
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    @patch('fedbiomed.node.node.Node.parser_task')     
    def test_task_manager_01_normal_case_scenario(self,
                                                  parser_task_patch,
                                                  send_msg_patch,
                                                  tasks_queue_get_patch,
                                                  round_patch):
        """Tests task_manager in the normal case scenario"""
        
        Round = MagicMock(run_model_training = None)
        #Round.run_model_training = MagicMock(return_value = None)
        # defining patchers
        parser_task_patch.return_value = None
        tasks_queue.return_value = None
        round_patch.return_value = None
        send_msg_patch.return_value = None
        tasks_queue_get_patch.return_value = None
        # defining arguments
        self.n1.rounds = [Round(),Round()]
        
        
        # action
        #thread_test = multiprocessing.Process(target=self.n1.task_manager
        #                                      )
        #thread_test.start()
        #self.n1.task_manager()
        #time.sleep(5)
        #print('here', thread_test.is_alive())
        #thread_test.join()
        #send_msg_patch.assert_called()
        #thread_test.terminate()
        
        # checks
        
        
        # close thread
        #thread_test.close()
    
    @patch('fedbiomed.common.messaging.Messaging.start')    
    def test_start_messaging_01(self,
                                msg_start_patch):
        """Tests `start_messaging` method (correct execution)"""
        # arguments
        block = True
        # action
        self.n1.start_messaging(block)
        
        # checks
        msg_start_patch.assert_called_once_with(block)
        
    @patch('fedbiomed.common.messaging.Messaging.send_error')
    def test_send_error(self, msg_send_error_patch):
        
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
if __name__ == '__main__': 
    unittest.main()
