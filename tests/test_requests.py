# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

from typing import Callable
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.monitor import Monitor
import unittest
from unittest.mock import patch


class TestRequest(unittest.TestCase):
    """ Test class for Request class """

    def setUp(self):

        """Setup mocks for Messaging class"""
        self.req_pathcer1 = patch('fedbiomed.common.messaging.Messaging.__init__')
        self.req_pathcer2 = patch('fedbiomed.common.messaging.Messaging.start')
        self.req_pathcer3 = patch('fedbiomed.common.messaging.Messaging.send_message')

        self.message_init           = self.req_pathcer1.start()
        self.message_start          = self.req_pathcer2.start()
        self.message_send           = self.req_pathcer3.start()

        self.message_init.return_value = None
        self.message_start.return_value = None
        self.message_send.return_value = None


    def tearDown(self):

        self.req_pathcer1.stop()
        self.req_pathcer1.stop()
        self.req_pathcer1.stop()

        pass


    @patch('fedbiomed.researcher.requests.Requests.get_responses')
    def test_list_function(self, request_get_response):

        # Test with single response database
        res = [
                {'node_id' : 'node-1',
                'researcher_id': 'r-xxx',
                'databases' : [
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'},
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'}
                ],
                'success' : True,
                'count' : 2,
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

        # Test with multople database response
        res = [
                {'node_id' : 'node-1',
                'researcher_id': 'r-xxx',
                'databases' : [
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'},
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'}
                ],
                'success' : True,
                'count' : 2,
                'command': 'list'
                },
                {'node_id' : 'node-2',
                'researcher_id': 'r-xxx',
                'databases' : [
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'},
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'}
                ],
                'success' : True,
                'count' : 2,
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
                {'node_id' : 'node',
                'researcher_id': 'r-xxx',
                'databases' : [
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'},
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'}
                ],
                'success' : True,
                'count' : 2,
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
                {'toto' : 'node',
                'researcher_id': 'r-xxx',
                'databases' : [
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'},
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'}
                ],
                'success' : True,
                'count' : 2,
                'command': 'list'
                },
                {'node_id' : 'node-2',
                'researcher_id': 'r-xxx',
                'databases' : [
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'},
                    {'data_type' : 'csv', 'tags': ['ss' , 'ss'], 'shape' : [1,2], 'name' : 'data'}
                ],
                'success' : True,
                'count' : 2,
                'command': 'list'
                }
            ]

        responses = Responses(res)
        request_get_response.return_value = responses

        try:
            req = Requests(nodes= ['node-1' , 'node-2'])
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
