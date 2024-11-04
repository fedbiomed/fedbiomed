import inspect
import os.path
import string
import random
import unittest
import time

from typing import Any, Dict
from unittest.mock import patch,PropertyMock, MagicMock, ANY
from threading import Semaphore
#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.constants import MessageType
from fedbiomed.common.message import Log, Scalar, SearchReply, SearchRequest, ErrorMessage, ApprovalReply

from fedbiomed.researcher.requests import (
    Requests,
    FederatedRequest,
    Request,
    RequestStatus,
    RequestPolicy,
    PolicyStatus,
    PolicyController,
    DiscardOnTimeout,
    StopOnTimeout,
    StopOnDisconnect,
    StopOnError,
    MessagesByNode)
from fedbiomed.researcher.monitor import Monitor

from fedbiomed.transport.node_agent import NodeAgent, NodeActiveStatus
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


    def setUp(self):
        """Setup Requests and patches for testing"""

        self.tp_abstract_patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())


        self.ssl_credentials_patch = patch('fedbiomed.researcher.requests._requests.SSLCredentials')

        self.grpc_server_patcher1 = patch('fedbiomed.transport.server.GrpcServer.__init__', autospec=True)
        self.grpc_server_patcher2 = patch('fedbiomed.transport.server.GrpcServer.start', autospec=True)
        self.grpc_server_patcher3 = patch('fedbiomed.transport.server.GrpcServer.send', autospec=True)
        self.grpc_server_patcher4 = patch('fedbiomed.transport.server.GrpcServer.broadcast', autospec=True)
        self.grpc_server_patcher5 = patch('fedbiomed.transport.server.GrpcServer.get_node', autospec=True)
        self.grpc_server_patcher6 = patch('fedbiomed.transport.server.GrpcServer.get_all_nodes', autospec=True)
        self.fed_req_enter_patcher1 = patch('fedbiomed.researcher.requests.FederatedRequest.__enter__', autospec=True)
        self.req_patcher4 = patch('fedbiomed.common.tasks_queue.TasksQueue.__init__')

        self.tp_abstract_patcher.start()

        self.ssl_credentials_mock = self.ssl_credentials_patch.start()
        self.grpc_server_init = self.grpc_server_patcher1.start()
        self.grpc_server_start = self.grpc_server_patcher2.start()
        self.grpc_server_send = self.grpc_server_patcher3.start()
        self.grpc_server_broadcast = self.grpc_server_patcher4.start()
        self.grpc_server_get_node = self.grpc_server_patcher5.start()
        self.grpc_server_get_all_nodes = self.grpc_server_patcher6.start()
        self.fed_req_enter = self.fed_req_enter_patcher1.start()
        self.task_queue_init = self.req_patcher4.start()

        self.grpc_server_init.return_value = None
        self.grpc_server_start.return_value = None
        self.grpc_server_send.return_value = None
        self.task_queue_init.return_value = None

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

        self.ssl_credentials_patch.stop()
        self.tp_abstract_patcher.stop()
        self.grpc_server_patcher1.stop()
        self.grpc_server_patcher2.stop()
        self.grpc_server_patcher3.stop()
        self.grpc_server_patcher4.stop()
        self.grpc_server_patcher5.stop()
        self.grpc_server_patcher6.stop()
        self.fed_req_enter_patcher1.stop()
        self.req_patcher4.stop()

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
                       'experiment_id': 'DummyExperimentID',
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
                      'msg': '{"message" : "Dummy Message"}'}
        self.requests.print_node_log_message(msg_logger)
        mock_logger_info.assert_called_once()

        with self.assertRaises(TypeError):
            # testing what is happening when no argument are provided
            self.requests.print_node_log_message()

    def test_request_04_send(self):
        """ Testing send message method of Request """

        sr = SearchRequest(
            researcher_id="researhcer-id",
            tags=["x"]
        )

        request_1 = self.requests.send(sr, None)
        self.grpc_server_get_all_nodes.assert_called_once()
        self.assertIsInstance(request_1, FederatedRequest)

        request_2 = self.requests.send({}, ['node-1'])
        self.grpc_server_get_node.assert_called_once_with(ANY, 'node-1')
        self.assertIsInstance(request_2, FederatedRequest)


    def test_request_08_ping_nodes(self):
        """ Testing ping method """

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
                'count': 2}),
            'node-2': SearchReply(**{
                'node_id': 'node-2',
                'researcher_id': 'r-xxx',
                'databases': [
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'},
                    {'data_type': 'csv', 'tags': ['ss', 'ss'], 'shape': [1, 2], 'name': 'data'}
                ],
                'count': 2})
        }

        fed_req = MagicMock()
        fed_req.replies.return_value = replies
        fed_req.errors.return_value = {'node-3': ErrorMessage(researcher_id="r",
                                                              node_id="node-3",
                                                              errnum="x",
                                                              extra_msg="x",
                                                            )}

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
                'count': 2}),
            'node-2': SearchReply(**{
                'node_id': 'node-2',
                'researcher_id': 'r-xxx',
                'databases': [],
                'count': 0})
        }

        fed_req = MagicMock()
        fed_req.replies.return_value = replies
        fed_req.errors.return_value = {'node-3': ErrorMessage(researcher_id="r",
                                                              node_id="node-3",
                                                              errnum="x",
                                                              extra_msg="x")}

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
    @patch('fedbiomed.researcher.requests._requests.import_class_object_from_file')
    @patch('fedbiomed.researcher.requests._requests.minify', return_value='hello')
    def test_request_13_training_plan_approve(self,
                                              mock_minify,
                                              mock_import,
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
                                                              extra_msg="x")}


        fed_req.replies.return_value = {"dummy-id-1": ApprovalReply(**{
            'node_id': 'dummy-id-1',
            'success': True,
            'training_plan_id': 'id-xx',
            'message': "hello",
            'researcher_id': "id",
            "status": True})}

        tp = MagicMock(spec=FakeTorchTrainingPlan2)
        tp.source.return_value = 'TP'

        mock_import.return_value = ('dummy', tp)
        result = self.requests.training_plan_approve(tp,
                                                     "test-training-plan-1",
                                                     nodes=["dummy-id-1"]
                                                     )

        keys = list(result.keys())
        self.assertTrue(len(keys), 1)
        self.assertTrue(result[keys[0]])


class TestRequest(unittest.TestCase):

    def setUp(self):
        self.message = MagicMock(spec=SearchRequest)
        self.node = MagicMock(spec=NodeAgent)
        self.sem_pending = MagicMock(spec=Semaphore)

        self.request  = Request(
            message = self.message,
            node=self.node,
            request_id = 'test-request-id',
            sem_pending = self.sem_pending
        )

        pass

    def test_01_request_send(self):

        # Test getter
        node = self.request.node
        self.assertEqual(node, self.request._node)


        self.request.send()
        self.node.send.assert_called_once_with(self.message, self.request.on_reply)
        self.assertEqual(self.request.status, RequestStatus.NO_REPLY_YET)


    def test_02_request_flush(self):

        self.request.flush(stopped = True)
        self.node.flush.assert_called_once_with(self.request._request_id, True)

    def test_03_request_on_reply(self):

        self.request.on_reply(self.message)
        self.assertEqual(self.request.reply, self.message)

        err_message = MagicMock(spec=ErrorMessage)
        self.request.on_reply(err_message)
        self.assertEqual(self.request.error, err_message)

    def test_04_request_has_finished(self):


        type(self.node).status = PropertyMock(return_value=NodeActiveStatus.DISCONNECTED)
        r = self.request.has_finished()
        self.assertTrue(r)

        type(self.node).status = PropertyMock(return_value=NodeActiveStatus.ACTIVE)
        r = self.request.has_finished()
        self.assertFalse(r)


        self.request.on_reply(self.message)
        r = self.request.has_finished()
        self.assertTrue(r)


class TestFederatedRequest(unittest.TestCase):

    def setUp(self):

        self.sem_patch = patch('fedbiomed.researcher.requests._requests.threading.Semaphore')
        self.sem_mock = self.sem_patch.start()

        self.policy_patch = patch('fedbiomed.researcher.requests._requests.PolicyController', autospec=True)
        self.policy_mock = self.policy_patch.start()

        self.message_1 = MagicMock(spec=SearchRequest)
        self.policy = MagicMock(spec=RequestPolicy)

        self.node_1 = MagicMock(spec=NodeAgent)
        type(self.node_1).id = PropertyMock(return_value='node-1')

        self.node_2 = MagicMock(spec=NodeAgent)
        type(self.node_2).id = PropertyMock(return_value='node-2')

        self.federated_request = FederatedRequest(
            message=self.message_1,
            nodes = [self.node_1, self.node_2],
            policy = [self.policy]
        )

    def tearDown(self):

        self.sem_patch.stop()
        self.policy_patch.stop()

    def test_01_federated_request_init(self):

        message = MessagesByNode(
            {'node-1' : self.message_1,
             'node-3' : self.message_1},  # prints warning message
        )

        r = FederatedRequest(
            message=message,
            nodes = [self.node_1, self.node_2],
            policy = [self.policy]
        )

        self.assertEqual(1, len(r.requests))


    def test_02_federeated_request_send(self):

        self.federated_request.send()

        self.node_1.send.assert_called_once_with(self.message_1, ANY)
        self.node_2.send.assert_called_once_with(self.message_1, ANY)


    def test_03_federated_request_wait(self):

        self.policy_mock.return_value.continue_all.side_effect = [ PolicyStatus.CONTINUE, PolicyStatus.COMPLETED]
        self.federated_request.wait()

    def test_04_federaeted_request_with_context_manager(self):

        self.policy_mock.return_value.continue_all.side_effect = [ PolicyStatus.CONTINUE, PolicyStatus.COMPLETED]

        with FederatedRequest(message=self.message_1,
                              nodes = [self.node_1, self.node_2],
                              policy = [self.policy]) as fed_req:
            self.assertEqual(2, len(fed_req.requests))
            self.assertEqual(0, len(fed_req.disconnected_requests()))
            self.assertEqual({}, fed_req.replies())
            self.assertEqual({}, fed_req.errors())



class TestRequestPolicy(unittest.TestCase):

    def setUp(self):

        self.req_1 = MagicMock(spec=Request)
        self.req_2 = MagicMock(spec=Request)
        self.pol = RequestPolicy()


    def test_01_request_policy_continue(self):

        self.req_1.has_finished.return_value = False
        self.req_2.has_finished.return_value = True

        r = self.pol.continue_([self.req_1, self.req_2])
        self.assertEqual(r, PolicyStatus.CONTINUE)


    def test_02_request_policy_stop(self):

        r = self.pol.stop(self.req_1)
        self.assertEqual(r, PolicyStatus.STOPPED)
        self.assertEqual(self.pol.status, PolicyStatus.STOPPED)
        self.assertEqual(self.pol.stop_caused_by, self.req_1)

    def test_03_request_policy_keep(self):
        r = self.pol.keep()
        self.assertEqual(r, PolicyStatus.CONTINUE)
        self.assertEqual(self.pol.status, PolicyStatus.CONTINUE)


    def test_04_request_policy_completed(self):
        r = self.pol.completed()
        self.assertEqual(r, PolicyStatus.COMPLETED)
        self.assertEqual(self.pol.status, PolicyStatus.COMPLETED)


class TestPolicyController(unittest.TestCase):

    def setUp(self):

        self.req_policy_patch = patch('fedbiomed.researcher.requests._policies.RequestPolicy')
        self.req_policy_mock = self.req_policy_patch.start()

        self.policy_1 = MagicMock()
        self.req_1 = MagicMock(spec=Request)
        self.req_2 = MagicMock(spec=Request)

        self.pol_cont = PolicyController(
            policies=[self.policy_1]
        )

    def tearDown(self):
        self.req_policy_patch.stop()


    def test_01_policy_controller_continue_all(self):

        self.req_policy_mock.return_value.continue_.return_value = PolicyStatus.CONTINUE
        self.policy_1.continue_.return_value = PolicyStatus.CONTINUE

        r = self.pol_cont.continue_all([self.req_1, self.req_1])
        self.assertEqual(r, PolicyStatus.CONTINUE)

    def test_02_policy_controller_has_stopped_any(self):

        type(self.policy_1).status = PropertyMock(return_value=PolicyStatus.STOPPED)
        r  = self.pol_cont.has_stopped_any()
        self.assertTrue(r)


    def test_03_policy_controller_(self):
        type(self.policy_1).status = PropertyMock(return_value=PolicyStatus.STOPPED)
        r = self.pol_cont.report()

        self.assertTrue(r[list(r.keys())[0]])


class TestPolicyImplementations(unittest.TestCase):

    def setUp(self):

        self.req = MagicMock(spec=Request)
        self.node = MagicMock()

        type(self.node).id = PropertyMock(return_value='node-1')
        type(self.req).node = PropertyMock(return_value=self.node)
        type(self.req).status = PropertyMock()
        type(self.req).error = PropertyMock()


    def test_01_discard_on_timeout(self):

        pol = DiscardOnTimeout(nodes=['node-1'], timeout=0.1)

        def time_out():
            time.sleep(1)
            return False

        self.req.has_finished.side_effect = time_out
        pol.continue_([self.req])
        pol.continue_([self.req])

    def test_02_stop_on_timeout(self):

        pol = StopOnTimeout(nodes=['node-1'], timeout=0.1)

        def time_out():
            time.sleep(1)
            return False

        self.req.has_finished.side_effect = time_out
        pol.continue_([self.req])
        r = pol.continue_([self.req])
        self.assertEqual(r, PolicyStatus.STOPPED)

    def test_01_stop_on_disconnect(self):

        pol = StopOnDisconnect(nodes=['node-1'], timeout=0)

        r = pol.continue_([self.req])
        self.assertEqual(r, PolicyStatus.CONTINUE)

        time.sleep(1)
        type(self.req).status = PropertyMock(return_value=RequestStatus.DISCONNECT)
        r = pol.continue_([self.req])
        self.assertEqual(r, PolicyStatus.STOPPED)

    def test_01_stop_on_error(self):

        pol = StopOnError(nodes=['node-1'])

        type(self.req).error = PropertyMock(return_value=False)
        r = pol.continue_([self.req])
        self.assertEqual(r, PolicyStatus.CONTINUE)
        type(self.req).error = PropertyMock(return_value={'error': True})
        r = pol.continue_([self.req])
        self.assertEqual(r, PolicyStatus.STOPPED)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
