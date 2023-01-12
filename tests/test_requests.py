import inspect
import os.path
import string
import random
import unittest

from typing import Any, Dict
from unittest.mock import patch, MagicMock

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################


from testsupport.fake_message import FakeMessages
from testsupport.fake_responses import FakeResponses

from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.common.training_plans import TorchTrainingPlan

from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.monitor import Monitor
from testsupport.base_fake_training_plan import BaseFakeTrainingPlan


# for test_request_13_model_approve
class TrainingPlanGood(BaseFakeTrainingPlan):
    pass


class TrainingPlanBad():
    pass


class TrainingPlanCannotInstanciate(BaseFakeTrainingPlan):
    def __init__(self):
        x = unknown_method()

    pass

class TrainingPlanCannotSave(BaseFakeTrainingPlan):
     def save_code(self, path: str):
         raise OSError


class TestRequests(ResearcherTestCase):
    """ Test class for Request class """

    @classmethod
    def setUpClass(cls) -> None:

        super().setUpClass()

        # defining common side effect functions
        def msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeMessages(msg)
            return fake_node_msg

        def responses_side_effect(data):
            fake_responses = FakeResponses(data)
            return fake_responses

        cls.msg_side_effect = msg_side_effect
        cls.responses_side_effect = responses_side_effect

    def setUp(self):
        """Setup Requests and patches for testing"""

        self.tp_abstract_patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())

        self.req_patcher1 = patch('fedbiomed.common.messaging.Messaging.__init__')
        self.req_patcher2 = patch('fedbiomed.common.messaging.Messaging.start')
        self.req_patcher3 = patch('fedbiomed.common.messaging.Messaging.send_message')
        self.req_patcher4 = patch('fedbiomed.common.tasks_queue.TasksQueue.__init__')
        self.req_patcher5 = patch('fedbiomed.common.message.ResearcherMessages.request_create')
        self.req_patcher6 = patch('fedbiomed.common.message.ResearcherMessages.reply_create')

        self.tp_abstract_patcher.start()

        self.message_init = self.req_patcher1.start()
        self.message_start = self.req_patcher2.start()
        self.message_send = self.req_patcher3.start()
        self.task_queue_init = self.req_patcher4.start()
        self.request_create = self.req_patcher5.start()
        self.reply_create = self.req_patcher6.start()

        self.message_init.return_value = None
        self.message_start.return_value = None
        self.message_send.return_value = None
        self.task_queue_init.return_value = None
        self.request_create.side_effect = TestRequests.msg_side_effect
        self.reply_create.side_effect = TestRequests.msg_side_effect

        # current directory
        self.cwd = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )

        # Remove singleton object and create fresh Request.
        # This is required to avoid attribute errors when there are
        # mocked Requests classes that come from other tests when running
        # tests on parallel with nosetests (did not worked in `tearDown`)
        if Requests in Requests._objects:
            del Requests._objects[Requests]
        self.requests = Requests()

    def tearDown(self):

        self.tp_abstract_patcher.stop()
        self.req_patcher1.stop()
        self.req_patcher2.stop()
        self.req_patcher3.stop()
        self.req_patcher4.stop()
        self.req_patcher5.stop()
        self.req_patcher6.stop()

        pass

    def test_request_01_constructor(self):
        """ Testing Request constructor """

        # Remove previous singleton instance
        if Requests in Requests._objects:
            del Requests._objects[Requests]

        # Build brand new reqeust by providing Messaging in advance
        messaging = Messaging()
        req_1 = Requests(mess=messaging)
        self.assertEqual(0, req_1._sequence, "Request is not properly initialized")
        self.assertEqual(None, req_1._monitor_message_callback, "Request is not properly initialized")
        self.assertEqual(messaging, req_1.messaging, "Request constructor didn't create proper Messaging")
        self.assertIsInstance(req_1.queue, TasksQueue, "Request constructor didn't create proper TasksQueue")

        # Remove previous singleton instance
        if Requests in Requests._objects:
            del Requests._objects[Requests]

        # Build new fresh reqeust
        req_2 = Requests(mess=None)
        self.assertEqual(0, req_1._sequence, "Request is not properly initialized")
        self.assertEqual(None, req_1._monitor_message_callback, "Request is not properly initialized")
        self.assertIsInstance(req_2.messaging, Messaging, "Request constructor didn't create proper Messaging")
        self.assertIsInstance(req_2.queue, TasksQueue, "Request constructor didn't create proper TasksQueue")

    def test_request_02_get_messaging(self):
        """ Testing the method `get_messaging`
            TODO: Update this part when refactoring getters and setter for reqeust
        """
        messaging = self.requests.get_messaging()
        self.assertIsInstance(messaging, Messaging, "get_messaging() does not return proper Messaging object")

    @patch('fedbiomed.researcher.requests.Requests.print_node_log_message')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.add')
    @patch('fedbiomed.common.logger.logger.error')
    def test_request_03_on_message(self,
                                   mock_logger_error,
                                   mock_task_add,
                                   mock_print_node_log_message):
        """ Testing on_message method """

        msg_logger = {'researcher_id': 'DummyID',
                      'node_id': 'DummyNodeID',
                      'level': 'critical',
                      'msg': '{"message" : "Dummy Message"}',
                      'command': 'log'}

        self.requests.on_message(msg_logger, topic='general/logger')
        # Check the method has been called
        mock_print_node_log_message.assert_called_once_with(msg_logger)

        msg_researcher_reply = {'researcher_id': 'DummyID',
                                'success': True,
                                'databases': [],
                                'count': 1,
                                'node_id': 'DummyNodeID',
                                'command': 'search'}

        self.requests.on_message(msg_researcher_reply, topic='general/researcher')
        # Get researcher reply for `assert_called_with`
        mock_task_add.assert_called_once_with(msg_researcher_reply)

        msg_monitor = {'researcher_id': 'DummyID',
                       'node_id': 'DummyNodeID',
                       'job_id': 'DummyJobID',
                       'key': 'loss',
                       'value': 12.23,
                       'epoch': 5,
                       'iteration': 15,
                       'command': 'add_scalar'}

        monitor_callback = MagicMock(return_value=None)
        # Add callback for monitoring
        self.requests.add_monitor_callback(monitor_callback)
        self.requests.on_message(msg_monitor, topic='general/monitoring')
        monitor_callback.assert_called_once_with(msg_monitor)

        # Test when the topic is unknown, it should call logger to log error
        self.requests.on_message(msg_monitor, topic='unknown/topic')
        mock_logger_error.assert_called_once()

        # Test invalid `on_message calls`
        with self.assertRaises(Exception):
            self.requests.on_message()
            self.requests.on_message(msg_monitor)
            self.requests.on_message(topic='unknown/topic')

    @patch('fedbiomed.common.logger.logger.info')
    def test_request_04_print_node_log_message(self, mock_logger_info):
        """ Testing printing log messages that comes from node """

        msg_logger = {'researcher_id': 'DummyID',
                      'node_id': 'DummyNodeID',
                      'level': 'critical',
                      'msg': '{"message" : "Dummy Message"}',
                      'command': 'log'}
        self.requests.print_node_log_message(msg_logger)
        mock_logger_info.assert_called_once()

        with self.assertRaises(TypeError):
            # testing what is happening when no argument are provided
            self.requests.print_node_log_message()

    @patch('fedbiomed.common.logger.logger.debug')
    def test_request_05_send_message(self, mock_logger_debug):
        """ Testing send message method of Request """

        self.requests.send_message({}, None)
        self.requests.send_message({}, 'NodeID')

        self.assertEqual(self.message_send.call_count, 2, 'Requests: send_message -> m.send_message called unexpected '
                                                          'number of times, expected: 2')
        self.assertEqual(mock_logger_debug.call_count, 2, 'Requests: send_message -> logger.debug called unexpected '
                                                          'number of times, expected: 2')

        # Test invalid call of send_message
        with self.assertRaises(TypeError):
            self.requests.send_message()

    @patch('fedbiomed.common.tasks_queue.TasksQueue.qsize')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.task_done')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.get')
    def test_request_06_get_messages(self,
                                     mock_task_get,
                                     mock_task_task_done,
                                     mock_task_qsize):
        """ Testing get_messages """

        mock_task_qsize.return_value = 1
        mock_task_task_done.return_value = None

        # Test with empty Task
        self.requests.get_messages(commands=['search'])
        mock_task_get.return_value = {}

        # Test with task
        data = {"command": 'train'}
        mock_task_get.return_value = data
        response = self.requests.get_messages(commands=['train'])

        # Check methods are called
        self.assertEqual(mock_task_task_done.call_count, 2, 'Requests:  get_messages -> queue.get called unexpected '
                                                            'number of times, expected: 2')
        self.assertEqual(mock_task_get.call_count, 2, 'Requests:  get_messages -> queue.get called unexpected number '
                                                      'of times, expected: 2')

        # Check result of the get_messages
        self.assertListEqual(response.data(), [data], 'get_messages result is not set correctly')

        # Test try/except block when .get() method exception
        mock_task_get.side_effect = FedbiomedTaskQueueError()

        resp1 = self.requests.get_messages(commands=['test-1'])
        # check if output is a `Responses` object
        self.assertIsInstance(resp1, Responses)

        # Test try/except block when .task_done() method raises exception
        mock_task_get.side_effect = None
        mock_task_task_done.side_effect = FedbiomedTaskQueueError
        resp2 = self.requests.get_messages(commands=['test-2'])
        # check if output is a `Responses` object
        self.assertIsInstance(resp2, Responses)

    @patch('fedbiomed.researcher.requests.Requests.get_messages')
    @patch('fedbiomed.researcher.responses.Responses')
    def test_request_07_get_responses(self,
                                      mock_responses_init,
                                      mock_get_messages):
        """ Testing get responses method """

        mock_responses_init.side_effect = TestRequests.responses_side_effect

        test_response = FakeResponses([{'command': 'test', 'success': True}])
        mock_get_messages.side_effect = [test_response,
                                         FakeResponses([])]

        responses_1 = self.requests.get_responses(look_for_commands='test', timeout=0.1)
        self.assertEqual(responses_1[0], test_response[0], 'Values of provided responses and values of result does not '
                                                           'match')

        # Test when `only_successful` or `while_responses` is False
        mock_get_messages.side_effect = [test_response,
                                         FakeResponses([])]

        responses_2 = self.requests.get_responses(look_for_commands='test', timeout=0.1, only_successful=False)
        self.assertEqual(responses_2[0], test_response[0], 'Values of provided responses and values of result does not '
                                                           'match')

        mock_get_messages.side_effect = [test_response,
                                         FakeResponses([])]

        responses_2bis = self.requests.get_responses(look_for_commands='test', timeout=0.1, while_responses=False)
        self.assertEqual(responses_2bis[0], test_response[0], 'Values of provided responses and values of result does not '
                                                           'match')

        mock_get_messages.side_effect = [Exception()]
        with self.assertRaises(Exception):
            self.requests.get_responses(look_for_commands='test', timeout=0.1, only_successful=False)

        # Get into Except block by providing incorrect message
        mock_get_messages.side_effect = [FakeResponses([{}])]
        responses_3 = self.requests.get_responses(look_for_commands='test', timeout=0.1)
        self.assertEqual(len(responses_3), 0, 'The length of responses are more than 0')

    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_request_08_ping_nodes(self, mock_get_responses):
        """ Testing ping method """

        mock_get_responses.return_value = [
            {'command': 'ping', 'node_id': 'dummy-id-1'},
            {'command': 'ping', 'node_id': 'dummy-id-2'},
        ]

        result = self.requests.ping_nodes()

        self.message_send.assert_called_once()
        self.assertEqual(result[0], 'dummy-id-1', 'Ping result does not contain provided node id `dummy-id-1`')
        self.assertEqual(result[1], 'dummy-id-2', 'Ping result does not contain provided node id `dummy-id-2`')

    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    @patch('fedbiomed.common.logger.logger.info')
    def test_reqeust_09_search(self,
                               mock_logger_info,
                               mock_get_responses):
        """ Testing search reqeust """

        mock_logger_info.return_value = None

        node_1 = {'node_id': 'node-1',
                  'researcher_id': 'r-xxx',
                  'databases': [
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                  ],
                  'success': True,
                  'count': 2,
                  'command': 'search'
                  }

        node_2 = {'node_id': 'node-2',
                  'researcher_id': 'r-xxx',
                  'databases': [
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                  ],
                  'success': True,
                  'count': 2,
                  'command': 'search'
                  }

        tags = ['test']

        # Test with single node without providing node ids
        mock_get_responses.return_value = FakeResponses([node_1])
        search_result = self.requests.search(tags=tags)
        self.assertTrue('node-1' in search_result, 'Requests search result does not contain `node-1`')
        self.assertEqual(mock_logger_info.call_count, 2, 'Requests: Search- > Logger called unexpected number of '
                                                         'times, expected: 2')

        # Test with multiple nodes by providing node ids
        mock_logger_info.reset_mock()
        mock_get_responses.return_value = FakeResponses([node_1, node_2])
        search_result_2 = self.requests.search(tags=tags, nodes=['node-1', 'node-2'])
        self.assertTrue('node-1' in search_result_2, 'Requests search result does not contain `node-1`')
        self.assertTrue('node-2' in search_result_2, 'Requests search result does not contain `node-2`')
        self.assertEqual(mock_logger_info.call_count, 3, 'Requests: Search- > Logger called unexpected number of '
                                                         'times, expected: 3')
        # Test with empty response
        mock_logger_info.reset_mock()
        mock_get_responses.return_value = FakeResponses([])
        search_result_3 = self.requests.search(tags=tags)
        self.assertDictEqual(search_result_3, {})
        self.assertEqual(mock_logger_info.call_count, 2, 'Requests: Search- > Logger called unexpected number of '
                                                         'times, expected: 2')

    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    @patch('tabulate.tabulate')
    @patch('fedbiomed.common.logger.logger.info')
    def test_request_10_list(self,
                             mock_logger_info,
                             mock_tabulate,
                             request_get_response):
        """ Testing list reqeust """

        mock_tabulate.return_value = 'Test'
        mock_logger_info.return_value = None

        # Test with single response database
        node_1 = {'node_id': 'node-1',
                  'researcher_id': 'r-xxx',
                  'databases': [
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                  ],
                  'success': True,
                  'count': 2,
                  'command': 'list'
                  }

        node_2 = {'node_id': 'node-2',
                  'researcher_id': 'r-xxx',
                  'databases': [
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                      {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                  ],
                  'success': True,
                  'count': 2,
                  'command': 'list'
                  }

        node_3 = {'node_id': 'node-3',
                  'researcher_id': 'r-xxx',
                  'databases': [],
                  'success': True,
                  'count': 2,
                  'command': 'list'
                  }

        request_get_response.return_value = FakeResponses([node_1])
        result = self.requests.list()
        self.assertTrue('node-1' in result, 'List result does not contain `node-1`')

        # Test with multiple nodes
        request_get_response.return_value = FakeResponses([node_1, node_2])
        result = self.requests.list()
        self.assertTrue('node-1' in result, 'List result does not contain `node-1` while testing multiple')
        self.assertTrue('node-2' in result, 'List result does not contain `node-1` while testing multiple')

        # Test verbosity
        request_get_response.return_value = FakeResponses([node_1, node_2])
        result = self.requests.list(verbose=True)
        self.assertTrue('node-1' in result, 'List result does not contain `node-1` while testing verbosity')
        self.assertTrue('node-2' in result, 'List result does not contain `node-1` while testing verbosity')
        self.assertEqual(mock_tabulate.call_count, 2, 'tabulate called unexpected number of times, expected: 2')
        # Logger will be called 5 times
        self.assertEqual(mock_logger_info.call_count, 5, 'logger.info called unexpected number of times, expected: 5')

        # Test verbosity with empty list of dataset
        mock_tabulate.reset_mock()
        mock_logger_info.reset_mock()
        request_get_response.return_value = FakeResponses([node_3])
        result = self.requests.list(verbose=True)
        mock_tabulate.assert_not_called()
        # Logger will be called 2 times
        self.assertEqual(mock_logger_info.call_count, 2, 'Logger called unexpected number of times, expected: 2')

        # Test by providing node_ids
        self.message_send.reset_mock()
        request_get_response.return_value = FakeResponses([node_1, node_2])
        result = self.requests.list(nodes=['node-1', 'node-2'])
        self.assertEqual(self.message_send.call_count, 2, 'send_message has been called times that are not equal to '
                                                          'expected')

        self.assertTrue('node-1' in result, 'List result does not contain correct values')
        self.assertTrue('node-2' in result, 'List result does not contain correct values')

    @patch('fedbiomed.researcher.monitor.Monitor.__init__')
    @patch('fedbiomed.researcher.monitor.Monitor.on_message_handler')
    @patch('fedbiomed.researcher.requests.Requests.add_monitor_callback')
    def test_request_11_add_monitor_callback(self,
                                             mock_monitor_callback,
                                             mock_monitor_message_handler,
                                             mock_monitor_init):

        """ Test adding monitor message callbacks """
        mock_monitor_init.return_value = None
        mock_monitor_message_handler.return_value = None
        monitor = Monitor()

        # Test adding monitor callback
        self.requests.add_monitor_callback(monitor.on_message_handler)
        mock_monitor_callback.assert_called_once_with(monitor.on_message_handler)

    @patch('fedbiomed.researcher.monitor.Monitor.__init__')
    @patch('fedbiomed.researcher.monitor.Monitor.on_message_handler')
    def test_request_12_remove_monitor_callback(self,
                                                mock_monitor_message_handler,
                                                mock_monitor_init
                                                ):
        """ Test removing monitor message callback """

        mock_monitor_init.return_value = None
        mock_monitor_message_handler.return_value = None
        monitor = Monitor()

        self.requests.add_monitor_callback(monitor.on_message_handler)
        self.requests.remove_monitor_callback()
        self.assertIsNone(self.requests._monitor_message_callback, "Monitor callback hasn't been removed")

    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_request_13_training_plan_approve(self,
                                      mock_get_responses,
                                      mock_upload_file):
        """ Testing training_plan_approve method """

        # ths should not work at all
        filename = 'X:/'
        for i in range(30):
            filename += random.choice(string.ascii_letters)

        result = self.requests.training_plan_approve(filename,
                                                     "this file does not exist",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})

        # this does not even send a message to the node
        # as this is not a python file
        filename = os.path.join(self.cwd,
                                "README.md")
        result = self.requests.training_plan_approve(filename,
                                                     "this is not a python file !!",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})

        # bad argument
        result = self.requests.training_plan_approve(filename,
                                                     "this is not a python file !!",
                                                     timeout=2,
                                                     nodes="and not a list of UUIDs"
                                                     )
        self.assertDictEqual(result, {})

        # model is not a TrainingPlan
        result = self.requests.training_plan_approve(TrainingPlanBad,
                                                     "not a training plan !!",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})

        # another wrong model
        result = self.requests.training_plan_approve(TrainingPlanCannotInstanciate,
                                                     "cannot instanciate",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})

       # model that cannot be save
        result = self.requests.training_plan_approve(TrainingPlanCannotSave,
                                                     "cannot save",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})


        # provide a real python file but do not get an answer before timeout
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        filename = os.path.join(self.cwd,
                                "test-training-plan",
                                "test-training-plan-1.txt")
        result = self.requests.training_plan_approve(filename,
                                                     "test-training-plan-1",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})

        # same, but with a node list -> different answer
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        filename = os.path.join(self.cwd,
                                "test-training-plan",
                                "test-training-plan-1.txt")
        result = self.requests.training_plan_approve(filename,
                                                     "test-training-plan-1",
                                                     timeout=2,
                                                     nodes=["dummy-id-1"]
                                                     )
        keys = list(result.keys())
        self.assertTrue(len(keys), 1)
        self.assertFalse(result[keys[0]])

        # same, but with a fake wrong sequence
        mock_get_responses.return_value = [
            {'command': 'approval',
             'node_id': 'dummy-id-1',
             'success': False,
             'sequence': -1}
        ]
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        filename = os.path.join(self.cwd,
                                "test-training-plan",
                                "test-training-plan-1.txt")
        result = self.requests.training_plan_approve(filename,
                                                     "test-training-plan-1",
                                                     timeout=2
                                                     )
        self.assertDictEqual(result, {})

        # sequence is OK, but simulated node did not download the file correctly
        self.requests._sequence = 12345
        mock_get_responses.return_value = [
            {'command': 'approval',
             'node_id': 'dummy-id-1',
             'success': False,
             'sequence': 12345}
        ]
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        filename = os.path.join(self.cwd,
                                "test-training-plan",
                                "test-training-plan-1.txt")
        result = self.requests.training_plan_approve(filename,
                                                     "test-training-plan-1",
                                                     timeout=2
                                                     )
        keys = list(result.keys())
        self.assertTrue(len(keys), 1)
        self.assertFalse(result[keys[0]])

        # this time everything is OK
        self.requests._sequence = 54321
        mock_get_responses.return_value = [
            {'command': 'approval',
             'node_id': 'dummy-id-1',
             'success': True,
             'sequence': 54321}
        ]
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        filename = os.path.join(self.cwd,
                                "test-training-plan",
                                "test-training-plan-1.txt")
        result = self.requests.training_plan_approve(filename,
                                                     "test-training-plan-1",
                                                     timeout=2
                                                     )
        keys = list(result.keys())
        self.assertTrue(result[keys[0]])

        # send a proper TrainingPlan
        model = TrainingPlanGood
        self.requests._sequence = 112233
        mock_get_responses.return_value = [
            {'command': 'approval',
             'node_id': 'dummy-id-1',
             'success': True,
             'sequence': 112233}
        ]
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        result = self.requests.training_plan_approve(model,
                                                     "model is a TrainingPlan subclass",
                                                     timeout=1
                                                     )
        keys = list(result.keys())
        self.assertTrue(result[keys[0]])

        # send an instance of TrainingPlan
        model = TrainingPlanGood()
        self.requests._sequence = 112233
        mock_get_responses.return_value = [
            {'command': 'approval',
             'node_id': 'dummy-id-1',
             'success': True,
             'sequence': 112233}
        ]
        mock_upload_file.return_value = {
            "node_id": "dummy-id-1",
            "url": "fake_url",
            "file": "fake_file",
            "success": True
        }
        result = self.requests.training_plan_approve(model,
                                                     "model is a TrainingPlan instance",
                                                     timeout=2
                                                     )
        keys = list(result.keys())
        self.assertTrue(result[keys[0]])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
