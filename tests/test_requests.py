import inspect
import os.path
import string
import random
import unittest

from typing import Any, Dict
from unittest.mock import patch, MagicMock, ANY

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################


from testsupport.fake_message import FakeMessages

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.constants import MessageType
from fedbiomed.common.message import Log, Scalar, SearchReply, SearchRequest, ErrorMessage, ApprovalReply

from fedbiomed.researcher.requests import Requests, FederatedRequest
from fedbiomed.researcher.monitor import Monitor
from testsupport.base_fake_training_plan import BaseFakeTrainingPlan
from testsupport.fake_training_plan import FakeTorchTrainingPlan2


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

        cls.msg_side_effect = msg_side_effect

    def setUp(self):
        """Setup Requests and patches for testing"""

        self.tp_abstract_patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())

        self.grpc_server_patcher1 = patch('fedbiomed.transport.server.GrpcServer.__init__', autospec=True)
        self.grpc_server_patcher2 = patch('fedbiomed.transport.server.GrpcServer.start', autospec=True)
        self.grpc_server_patcher3 = patch('fedbiomed.transport.server.GrpcServer.send', autospec=True)
        self.grpc_server_patcher4 = patch('fedbiomed.transport.server.GrpcServer.broadcast', autospec=True)
        self.grpc_server_patcher5 = patch('fedbiomed.transport.server.GrpcServer.get_node', autospec=True)
        self.grpc_server_patcher6 = patch('fedbiomed.transport.server.GrpcServer.get_all_nodes', autospec=True)
        self.fed_req_enter_patcher1 = patch('fedbiomed.researcher.requests.FederatedRequest.__enter__', autospec=True)
        self.req_patcher4 = patch('fedbiomed.common.tasks_queue.TasksQueue.__init__')
        self.req_patcher5 = patch('fedbiomed.common.message.ResearcherMessages.format_outgoing_message')
        self.req_patcher6 = patch('fedbiomed.common.message.ResearcherMessages.format_incoming_message')

        self.tp_abstract_patcher.start()

        self.grpc_server_init = self.grpc_server_patcher1.start()
        self.grpc_server_start = self.grpc_server_patcher2.start()
        self.grpc_server_send = self.grpc_server_patcher3.start()
        self.grpc_server_broadcast = self.grpc_server_patcher4.start()
        self.grpc_server_get_node = self.grpc_server_patcher5.start()
        self.grpc_server_get_all_nodes = self.grpc_server_patcher6.start()
        self.fed_req_enter = self.fed_req_enter_patcher1.start()
        self.task_queue_init = self.req_patcher4.start()
        self.format_outgoing_message = self.req_patcher5.start()
        self.format_incoming_message = self.req_patcher6.start()

        self.grpc_server_init.return_value = None
        self.grpc_server_start.return_value = None
        self.grpc_server_send.return_value = None
        self.task_queue_init.return_value = None
        self.format_outgoing_message.side_effect = TestRequests.msg_side_effect
        self.format_incoming_message.side_effect = TestRequests.msg_side_effect

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
        self.grpc_server_patcher1.stop()
        self.grpc_server_patcher2.stop()
        self.grpc_server_patcher3.stop()
        self.grpc_server_patcher4.stop()
        self.grpc_server_patcher5.stop()
        self.grpc_server_patcher6.stop()
        self.fed_req_enter_patcher1.stop()
        self.req_patcher4.stop()
        self.req_patcher5.stop()
        self.req_patcher6.stop()

        pass

    def test_request_01_constructor(self):
        """ Testing Request constructor """

        # Remove previous singleton instance
        if Requests in Requests._objects:
            del Requests._objects[Requests]


        req_1 = Requests()
        self.assertEqual(None, req_1._monitor_message_callback, "Request is not properly initialized")

        # Remove previous singleton instance
        if Requests in Requests._objects:
            del Requests._objects[Requests]

        # Build new fresh requests
        req_2 = Requests()
        self.assertEqual(None, req_2._monitor_message_callback, "Request is not properly initialized")

    @patch('fedbiomed.researcher.requests.Requests.print_node_log_message')
    @patch('fedbiomed.common.logger.logger.error')
    def test_request_02_on_message(self,
                                   mock_logger_error,
                                   mock_print_node_log_message):
        """ Testing on_message method """

        msg_logger = {'node_id': 'DummyNodeID',
                      'level': 'critical',
                      'msg': '{"message" : "Dummy Message"}'
                      }
        msg_logger = Log(**msg_logger)


        self.requests.on_message(msg_logger, type_=MessageType.LOG)
        # Check the method has been called
        mock_print_node_log_message.assert_called_once_with(msg_logger.get_dict())

        msg_monitor = {'node_id': 'DummyNodeID',
                       'job_id': 'DummyJobID',
                       'metric': {"loss": 12},
                       'train': True,
                       'test': False,
                       'test_on_local_updates': True,
                       'test_on_global_updates': False,
                       'total_samples': 15,
                       'batch_samples': 15,
                       'num_batches': 15,
                       'iteration': 1,
                       'epoch': 5,
                       'iteration': 15}
        
        msg_monitor = Scalar(**msg_monitor)
        monitor_callback = MagicMock(return_value=None)
        # Add callback for monitoring
        self.requests.add_monitor_callback(monitor_callback)
        self.requests.on_message(msg_monitor, type_=MessageType.SCALAR)
        monitor_callback.assert_called_once_with(msg_monitor.get_dict())

        # Test when the topic is unknown, it should call logger to log error
        self.requests.on_message(msg_monitor, type_='unknown/topic')
        mock_logger_error.assert_called_once()

        # Test invalid `on_message calls`
        with self.assertRaises(Exception):
            self.requests.on_message()
            self.requests.on_message(msg_monitor)
            self.requests.on_message(type_=MessageType.SCALAR)

    @patch('fedbiomed.common.logger.logger.info')
    def test_request_03_print_node_log_message(self, mock_logger_info):
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

    def test_request_04_send(self):
        """ Testing send message method of Request """

        sr = SearchRequest(
            researcher_id="researhcer-id",
            tags=["x"],
            command="search"
        )

        request_1 = self.requests.send(sr, None)
        self.grpc_server_get_all_nodes.assert_called_once()
        self.assertIsInstance(request_1, FederatedRequest)

        request_2 = self.requests.send({}, ['node-1'])
        self.grpc_server_get_node.assert_called_once_with(ANY, 'node-1')
        self.assertIsInstance(request_2, FederatedRequest)


    def test_request_08_ping_nodes(self):
        """ Testing ping method """

        self.req_patcher5.stop()
        self.requests.ping_nodes()
        self.fed_req_enter.assert_called_once()

    @patch('fedbiomed.researcher.requests.Requests.send')
    def test_request_09_search(self,
                               send):
        """ Testing search request """

        replies = {
            'node-1': SearchReply(**{
                'node_id': 'node-1',
                'researcher_id': 'r-xxx',
                'databases': [
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                ],
                'success': True,
                'count': 2,
                'command': 'search'}),
            'node-2': SearchReply(**{
                'node_id': 'node-2',
                'researcher_id': 'r-xxx',
                'databases': [
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                ],
                'success': True,
                'count': 2,
                'command': 'search'})
        }

        fed_req = MagicMock()
        fed_req.replies.return_value = replies
        fed_req.errors.return_value = {'node-3': ErrorMessage(researcher_id="r", 
                                                              node_id="node-3",  
                                                              errnum="x", 
                                                              extra_msg="x", 
                                                              command="err" )}

        send.return_value.__enter__.return_value = fed_req
        tags = ['test']

        # Test with single node without providing node ids
        search_result = self.requests.search(tags=tags)
        self.assertTrue('node-2' in search_result.keys())
        self.assertTrue('node-1' in search_result.keys())
     


    @patch('tabulate.tabulate')
    @patch('fedbiomed.researcher.requests.Requests.send')
    def test_request_10_list(self,
                             send,
                             mock_tabulate):
        """ Testing list reqeust """

        mock_tabulate.return_value = 'Test'

        # Test with single response database

        replies = {
            'node-1': SearchReply(**{
                'node_id': 'node-1',
                'researcher_id': 'r-xxx',
                'databases': [
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                ],
                'success': True,
                'count': 2,
                'command': 'list'}),
            'node-2': SearchReply(**{
                'node_id': 'node-2',
                'researcher_id': 'r-xxx',
                'databases': [],
                'success': True,
                'count': 0,
                'command': 'list'})
        }

        fed_req = MagicMock()
        fed_req.replies.return_value = replies
        fed_req.errors.return_value = {'node-3': ErrorMessage(researcher_id="r", 
                                                              node_id="node-3",  
                                                              errnum="x", 
                                                              extra_msg="x", 
                                                              command="err" )}

        send.return_value.__enter__.return_value = fed_req
        r = self.requests.list()
        self.assertTrue('node-1' in r )

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

    @patch('fedbiomed.researcher.requests.Requests.send')
    def test_request_13_training_plan_approve(self,
                                              send):
        """ Testing training_plan_approve method """

        # model is not a TrainingPlan
        result = self.requests.training_plan_approve(TrainingPlanBad,
                                                     "not a training plan !!"
                                                     )
        self.assertDictEqual(result, {})


        fed_req = MagicMock()
        send.return_value.__enter__.return_value = fed_req
        fed_req.errors.return_value = {'node-3': ErrorMessage(researcher_id="r", 
                                                              node_id="node-3",  
                                                              errnum="x", 
                                                              extra_msg="x", 
                                                              command="err" )}


        fed_req.replies.return_value = {"dummy-id-1": ApprovalReply(**{
            'command': 'approval',
            'node_id': 'dummy-id-1',
            'success': True,
            'message': "hello",
            'researcher_id': "id",
            "status": True})}

        result = self.requests.training_plan_approve(FakeTorchTrainingPlan2,
                                                     "test-training-plan-1",
                                                     nodes=["dummy-id-1"]
                                                     )

        keys = list(result.keys())
        self.assertTrue(len(keys), 1)
        self.assertTrue(result[keys[0]])



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
