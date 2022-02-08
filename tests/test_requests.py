# Managing NODE, RESEARCHER environ mock before running tests
import fedbiomed.researcher.requests
from testsupport.delete_environ import delete_environ

# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
import json
from typing import Callable
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.message import ResearcherMessages
import unittest
from unittest.mock import patch, MagicMock


class TestRequest(unittest.TestCase):
    """ Test class for Request class """

    class FaceMonitorCallback():
        pass

    def setUp(self):

        """Setup mocks for Messaging class"""
        self.req_pathcer1 = patch('fedbiomed.common.messaging.Messaging.__init__')
        self.req_pathcer2 = patch('fedbiomed.common.messaging.Messaging.start')
        self.req_pathcer3 = patch('fedbiomed.common.messaging.Messaging.send_message')

        self.message_init = self.req_pathcer1.start()
        self.message_start = self.req_pathcer2.start()
        self.message_send = self.req_pathcer3.start()

        self.message_init.return_value = None
        self.message_start.return_value = None
        self.message_send.return_value = None

    def tearDown(self):

        self.req_pathcer1.stop()
        self.req_pathcer2.stop()
        self.req_pathcer3.stop()

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

        # Remove previous singleton instance
        if Requests in Requests._objects:
            del Requests._objects[Requests]

        # Build new fresh reqeust
        req_2 = Requests(mess=None)
        self.assertEqual(0, req_1._sequence, "Request is not properly initialized")
        self.assertEqual(None, req_1._monitor_message_callback, "Request is not properly initialized")
        self.assertIsInstance(req_2.messaging, Messaging, "Request constructor didn't create proper Messaging")

    def test_request_02_get_messaging(self):
        """ Testing the method `get_messaging`
            TODO: Update this part when refactoring getters and setter for reqeust
        """
        req = Requests()
        messaging = req.get_messaging()
        self.assertIsInstance(messaging, Messaging, "get_messaging() does not return proper Messaging object")

    @patch('fedbiomed.researcher.requests.Requests.print_node_log_message')
    @patch('fedbiomed.common.tasks_queue.TasksQueue.add')
    @patch('fedbiomed.common.logger.logger.error')
    def test_request_03_on_message(self,
                                   mock_logger_error,
                                   mock_task_add,
                                   mock_print_node_log_message):
        """ Testing different scenarios for on_message methods """

        # Build reqeust
        req = Requests()

        msg_logger = {'researcher_id': 'DummyID',
                      'node_id': 'DummyNodeID',
                      'level': 'critical',
                      'msg': '{"message" : "Dummy Message"}',
                      'command': 'log'}

        # Get researcher reply for `assert_called_with`
        reply_logger = ResearcherMessages.reply_create(msg_logger).get_dict()
        req.on_message(msg_logger, topic='general/logger')

        # Check the method has been called
        mock_print_node_log_message.assert_called_once_with(reply_logger)

        msg_researcher_reply = {'researcher_id': 'DummyID',
                                'success': True,
                                'databases': [],
                                'count': 1,
                                'node_id': 'DummyNodeID',
                                'command': 'search'}

        req.on_message(msg_researcher_reply, topic='general/researcher')
        # Get researcher reply for `assert_called_with`
        reply_researcher = ResearcherMessages.reply_create(msg_researcher_reply).get_dict()
        mock_task_add.assert_called_once_with(reply_researcher)

        msg_monitor = {'researcher_id': 'DummyID',
                       'node_id': 'DummyNodeID',
                       'job_id': 'DummyJobID',
                       'key': 'loss',
                       'value': 12.23,
                       'epoch': 5,
                       'iteration': 15,
                       'command': 'add_scalar'}

        monitor_callback = MagicMock(return_value=None)

        # Get researcher reply for `assert_called_with`
        reply_monitor = ResearcherMessages.reply_create(msg_monitor).get_dict()
        # Add callback for monitoring
        req.add_monitor_callback(monitor_callback)
        req.on_message(msg_monitor, topic='general/monitoring')
        monitor_callback.assert_called_once_with(reply_monitor)

        # Test when the topic is unkown, it should call logger to log error
        req.on_message(msg_monitor, topic='unknown/topic')
        mock_logger_error.assert_called_once()

    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_request_04_list(self, request_get_response):

        # Test with single response database
        res = [
            {'node_id': 'node-1',
             'researcher_id': 'r-xxx',
             'databases': [
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
             ],
             'success': True,
             'count': 2,
             'command': 'list'
             }
        ]

        responses = Responses(res)
        request_get_response.return_value = responses
        try:
            req = Requests()
            result = req.list()
        except:
            self.assertTrue(False, 'List method failed even data is okay')

        self.assertIsInstance(result, object)

        # Test with multiple database response
        res = [
            {'node_id': 'node-1',
             'researcher_id': 'r-xxx',
             'databases': [
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
             ],
             'success': True,
             'count': 2,
             'command': 'list'
             },
            {'node_id': 'node-2',
             'researcher_id': 'r-xxx',
             'databases': [
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
             ],
             'success': True,
             'count': 2,
             'command': 'list'
             }
        ]

        responses = Responses(res)
        request_get_response.return_value = responses

        try:
            req = Requests()
            result = req.list()
        except:
            self.assertTrue(False, 'List method failed even data is okay')

        self.assertIsInstance(result, object)

        # Test with verbose mode
        res = [
            {'node_id': 'node',
             'researcher_id': 'r-xxx',
             'databases': [
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
             ],
             'success': True,
             'count': 2,
             'command': 'list'
             },
        ]

        responses = Responses(res)
        request_get_response.return_value = responses

        try:
            req = Requests(verbose=True)
            result = req.list()
        except:
            self.assertTrue(False, 'List method failed even data is okay')

        self.assertIsInstance(result, object)
        # Test with node ids
        res = [
            {'toto': 'node',
             'researcher_id': 'r-xxx',
             'databases': [
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
             ],
             'success': True,
             'count': 2,
             'command': 'list'
             },
            {'node_id': 'node-2',
             'researcher_id': 'r-xxx',
             'databases': [
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                 {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
             ],
             'success': True,
             'count': 2,
             'command': 'list'
             }
        ]

        responses = Responses(res)
        request_get_response.return_value = responses

        try:
            req = Requests(nodes=['node-1', 'node-2'])
            result = req.list()
        except:
            self.assertTrue(False, 'List method failed even data is okay')

        self.assertIsInstance(result, object)

    def test_add_remove_monitor_callback(self):

        """ Test adding and removing monitor message callbacks """

        req = Requests()
        monitor = Monitor()

        # Test adding monitor callback
        req.add_monitor_callback(monitor.on_message_handler)
        self.assertIsInstance(req._monitor_message_callback, Callable, "Monitor callback hasn't been added properly")

        # Test removing monitor callback
        req.remove_monitor_callback()
        self.assertIsNone(req._monitor_message_callback, "Monitor callback han't been removed")


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
