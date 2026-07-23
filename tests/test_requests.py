import os.path
import tempfile
import time
from threading import Semaphore
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import pytest
from testsupport.fake_training_plan import FakeTorchTrainingPlan2

from fedbiomed.common.certificate_manager import (
    CertificateManager,
    TrustedCertificateBundle,
)
from fedbiomed.common.constants import ComponentType, MessageType
from fedbiomed.common.exceptions import FedbiomedCertificateError
from fedbiomed.common.message import (
    ApprovalReply,
    ErrorMessage,
    Log,
    Scalar,
    SearchReply,
    SearchRequest,
)
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.researcher.config import config
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.requests import (
    DiscardOnTimeout,
    FederatedRequest,
    MessagesByNode,
    PolicyController,
    PolicyStatus,
    Request,
    RequestPolicy,
    Requests,
    RequestStatus,
    StopOnDisconnect,
    StopOnError,
    StopOnTimeout,
)
from fedbiomed.transport.node_agent import NodeActiveStatus, NodeAgent


# for test_request_training_plan_approve
class TrainingPlanBad:
    pass


def _search_reply(node_id, node_name, count):
    database = {
        "data_type": "csv",
        "tags": ["ss", "ss"],
        "shape": [1, 2],
        "name": "data",
    }
    return SearchReply(
        node_id=node_id,
        node_name=node_name,
        researcher_id="r-xxx",
        databases=[database] * count,
        count=count,
    )


def _error_message(node_id, node_name):
    return ErrorMessage(
        researcher_id="r",
        node_id=node_id,
        node_name=node_name,
        errnum="x",
        extra_msg="x",
    )


# -----------------------------------------------------------------------------
# Requests
# -----------------------------------------------------------------------------


@pytest.fixture
def requests_env():
    """Requests singleton with the gRPC server stack patched out."""
    with (
        patch.multiple(TorchTrainingPlan, __abstractmethods__=set()),
        patch("fedbiomed.researcher.requests._requests.SSLCredentials"),
        patch(
            "fedbiomed.transport.server.GrpcServer.__init__",
            autospec=True,
            return_value=None,
        ),
        patch(
            "fedbiomed.transport.server.GrpcServer.start",
            autospec=True,
            return_value=None,
        ),
        patch(
            "fedbiomed.transport.server.GrpcServer.send",
            autospec=True,
            return_value=None,
        ),
        patch("fedbiomed.transport.server.GrpcServer.broadcast", autospec=True),
        patch(
            "fedbiomed.transport.server.GrpcServer.get_node", autospec=True
        ) as grpc_server_get_node,
        patch(
            "fedbiomed.transport.server.GrpcServer.get_all_nodes", autospec=True
        ) as grpc_server_get_all_nodes,
        patch(
            "fedbiomed.researcher.requests.FederatedRequest.__enter__", autospec=True
        ) as fed_req_enter,
        patch("fedbiomed.common.tasks_queue.TasksQueue.__init__", return_value=None),
    ):
        # Remove singleton object and create fresh Request, to avoid attribute
        # errors from mocked Requests instances leaked by other test modules.
        if Requests in Requests._objects:
            del Requests._objects[Requests]

        temp_dir = tempfile.TemporaryDirectory()
        config.load(root=temp_dir.name)
        yield SimpleNamespace(
            requests=Requests(config=config),
            grpc_server_get_node=grpc_server_get_node,
            grpc_server_get_all_nodes=grpc_server_get_all_nodes,
            fed_req_enter=fed_req_enter,
            temp_dir=temp_dir,
        )
        temp_dir.cleanup()


def test_request_constructor(requests_env):
    """A fresh Requests singleton after deletion is properly initialized"""

    # Remove previous singleton instance
    if Requests in Requests._objects:
        del Requests._objects[Requests]

    req = Requests(config)
    assert req._monitor_message_callbacks == {}, "Request is not properly initialized"


@patch("fedbiomed.researcher.requests.Requests.print_node_log_message")
@patch("fedbiomed.common.logger.logger.error")
def test_request_on_message(
    mock_logger_error, mock_print_node_log_message, requests_env
):
    """Testing on_message method"""
    requests = requests_env.requests

    msg_logger = {
        "node_id": "DummyNodeID",
        "level": "critical",
        "msg": '{"message" : "Dummy Message"}',
    }
    msg_logger = Log(**msg_logger)

    requests.on_message(msg_logger, type_=MessageType.LOG)
    # Check the method has been called
    mock_print_node_log_message.assert_called_once_with(msg_logger.get_dict())

    experiment_id = "DummyExperimentID"
    msg_monitor = {
        "node_id": "DummyNodeID",
        "node_name": "DummyNodeName",
        "experiment_id": experiment_id,
        "metric": {"loss": 12},
        "train": True,
        "test": False,
        "test_on_local_updates": True,
        "test_on_global_updates": False,
        "total_samples": 15,
        "batch_samples": 15,
        "num_batches": 15,
        "epoch": 5,
        "iteration": 15,
    }

    msg_monitor = Scalar(**msg_monitor)
    monitor_callback = MagicMock(return_value=None)
    # Add callback for monitoring
    requests.add_monitor_callback(experiment_id, monitor_callback)
    requests.on_message(msg_monitor, type_=MessageType.SCALAR)
    monitor_callback.assert_called_once_with(msg_monitor.get_dict())

    # Test with other experiment id, the callback should not be called
    monitor_callback.reset_mock()
    msg_monitor.experiment_id = "AnotherExperimentID"
    requests.on_message(msg_monitor, type_=MessageType.SCALAR)
    monitor_callback.assert_not_called()

    # Test when the topic is unknown, it should call logger to log error
    requests.on_message(msg_monitor, type_="unknown/topic")
    mock_logger_error.assert_called_once()

    # Test invalid `on_message calls`
    with pytest.raises(TypeError):
        requests.on_message()

    with pytest.raises(TypeError):
        requests.on_message(msg_monitor)

    with pytest.raises(TypeError):
        requests.on_message(type_=MessageType.SCALAR)


@patch("fedbiomed.common.logger.logger.info")
def test_request_print_node_log_message(mock_logger_info, requests_env):
    """Testing printing log messages that comes from node"""

    msg_logger = {
        "researcher_id": "DummyID",
        "node_id": "DummyNodeID",
        "level": "critical",
        "msg": '{"message" : "Dummy Message"}',
    }
    requests_env.requests.print_node_log_message(msg_logger)
    mock_logger_info.assert_called_once()

    with pytest.raises(TypeError):
        # testing what is happening when no argument are provided
        requests_env.requests.print_node_log_message()


def test_request_send(requests_env):
    """Testing send message method of Request"""

    sr = SearchRequest(researcher_id="researcher-id", tags=["x"])

    request_1 = requests_env.requests.send(sr, None)
    requests_env.grpc_server_get_all_nodes.assert_called_once()
    assert isinstance(request_1, FederatedRequest)

    request_2 = requests_env.requests.send({}, ["node-1"])
    requests_env.grpc_server_get_node.assert_called_once_with(ANY, "node-1")
    assert isinstance(request_2, FederatedRequest)


def test_request_ping_nodes(requests_env):
    """Testing ping method"""

    requests_env.requests.ping_nodes()
    requests_env.fed_req_enter.assert_called_once()


@patch("fedbiomed.researcher.requests.Requests.send")
def test_request_search(send, requests_env):
    """Testing search request"""

    fed_req = MagicMock()
    fed_req.replies.return_value = {
        "node-1": _search_reply("node-1", "test-node1", count=2),
        "node-2": _search_reply("node-2", "test-node2", count=2),
    }
    fed_req.errors.return_value = {"node-3": _error_message("node-3", "test-node3")}

    send.return_value.__enter__.return_value = fed_req

    # Test with single node without providing node ids
    search_result = requests_env.requests.search(tags=["test"])
    assert "node-2" in search_result.keys()
    assert "node-1" in search_result.keys()


@patch("tabulate.tabulate")
@patch("fedbiomed.researcher.requests.Requests.send")
def test_request_list(send, mock_tabulate, requests_env):
    """Testing list request"""

    mock_tabulate.return_value = "Test"

    fed_req = MagicMock()
    # One node with databases, one without
    fed_req.replies.return_value = {
        "node-1": _search_reply("node-1", "test-node1", count=2),
        "node-2": _search_reply("node-2", "test-node2", count=0),
    }
    fed_req.errors.return_value = {"node-3": _error_message("node-3", "test-node-3")}

    send.return_value.__enter__.return_value = fed_req
    r = requests_env.requests.list()
    assert "node-1" in r


@patch("fedbiomed.researcher.monitor.Monitor.__init__")
@patch("fedbiomed.researcher.monitor.Monitor.on_message_handler")
def test_request_add_monitor_callback(
    mock_monitor_message_handler, mock_monitor_init, requests_env
):
    """Test adding monitor message callbacks"""
    mock_monitor_init.return_value = None
    mock_monitor_message_handler.return_value = None
    monitor = Monitor(results_dir=requests_env.temp_dir.name)
    experiment_id = "dummy-experiment-id"

    # Test adding monitor callback
    requests_env.requests.add_monitor_callback(
        experiment_id, monitor.on_message_handler
    )
    assert list(requests_env.requests._monitor_message_callbacks.keys()) == [
        experiment_id
    ]

    # Test adding monitor callback again
    requests_env.requests.add_monitor_callback(
        experiment_id, monitor.on_message_handler
    )
    assert list(requests_env.requests._monitor_message_callbacks.keys()) == [
        experiment_id
    ]


@patch("fedbiomed.researcher.monitor.Monitor.__init__")
@patch("fedbiomed.researcher.monitor.Monitor.on_message_handler")
def test_request_remove_monitor_callback(
    mock_monitor_message_handler, mock_monitor_init, requests_env
):
    """Test removing monitor message callback"""

    mock_monitor_init.return_value = None
    mock_monitor_message_handler.return_value = None
    monitor = Monitor(results_dir=requests_env.temp_dir.name)
    experiment_id = "dummy-experiment-id"

    requests_env.requests.add_monitor_callback(
        experiment_id, monitor.on_message_handler
    )
    requests_env.requests.remove_monitor_callback(experiment_id)
    assert requests_env.requests._monitor_message_callbacks == {}, (
        "Monitor callback hasn't been removed"
    )


@patch("fedbiomed.researcher.requests.Requests.send")
@patch("fedbiomed.researcher.requests._requests.import_class_object_from_file")
@patch("fedbiomed.researcher.requests._requests.minify", return_value="hello")
def test_request_training_plan_approve(mock_minify, mock_import, send, requests_env):
    """Testing training_plan_approve method"""

    # model is not a TrainingPlan
    result = requests_env.requests.training_plan_approve(
        TrainingPlanBad, "not a training plan !!"
    )
    assert result == {}

    fed_req = MagicMock()
    send.return_value.__enter__.return_value = fed_req
    fed_req.errors.return_value = {"node-3": _error_message("node-3", "test-node-3")}

    fed_req.replies.return_value = {
        "dummy-id-1": ApprovalReply(
            **{
                "node_id": "dummy-id-1",
                "node_name": "test-node-1",
                "success": True,
                "training_plan_id": "id-xx",
                "message": "hello",
                "researcher_id": "id",
                "status": True,
            }
        )
    }

    tp = MagicMock(spec=FakeTorchTrainingPlan2)
    tp.source.return_value = "TP"

    mock_import.return_value = ("dummy", tp)
    result = requests_env.requests.training_plan_approve(
        tp, "test-training-plan-1", nodes=["dummy-id-1"]
    )

    keys = list(result.keys())
    assert len(keys) == 1
    assert result[keys[0]]


# -----------------------------------------------------------------------------
# Request
# -----------------------------------------------------------------------------


@pytest.fixture
def request_env():
    message = MagicMock(spec=SearchRequest)
    node = MagicMock(spec=NodeAgent)
    return SimpleNamespace(
        message=message,
        node=node,
        request=Request(
            message=message,
            node=node,
            request_id="test-request-id",
            sem_pending=MagicMock(spec=Semaphore),
        ),
    )


def test_request_send_single(request_env):
    # Test getter
    assert request_env.request.node == request_env.request._node

    request_env.request.send()
    request_env.node.send.assert_called_once_with(
        request_env.message, request_env.request.on_reply
    )
    assert request_env.request.status == RequestStatus.NO_REPLY_YET


def test_request_flush(request_env):
    request_env.request.flush(stopped=True)
    request_env.node.flush.assert_called_once_with(
        request_env.request._request_id, True
    )


def test_request_on_reply(request_env):
    request_env.request.on_reply(request_env.message)
    assert request_env.request.reply == request_env.message

    err_message = MagicMock(spec=ErrorMessage)
    request_env.request.on_reply(err_message)
    assert request_env.request.error == err_message


def test_request_has_finished(request_env):
    type(request_env.node).status = PropertyMock(
        return_value=NodeActiveStatus.DISCONNECTED
    )
    assert request_env.request.has_finished()

    type(request_env.node).status = PropertyMock(return_value=NodeActiveStatus.ACTIVE)
    assert not request_env.request.has_finished()

    request_env.request.on_reply(request_env.message)
    assert request_env.request.has_finished()


# -----------------------------------------------------------------------------
# FederatedRequest
# -----------------------------------------------------------------------------


@pytest.fixture
def federated_request_env():
    with (
        patch("fedbiomed.researcher.requests._requests.threading.Semaphore"),
        patch(
            "fedbiomed.researcher.requests._requests.PolicyController", autospec=True
        ) as policy_mock,
    ):
        message_1 = MagicMock(spec=SearchRequest)
        policy = MagicMock(spec=RequestPolicy)

        node_1 = MagicMock(spec=NodeAgent)
        type(node_1).id = PropertyMock(return_value="node-1")

        node_2 = MagicMock(spec=NodeAgent)
        type(node_2).id = PropertyMock(return_value="node-2")

        yield SimpleNamespace(
            policy_mock=policy_mock,
            message_1=message_1,
            policy=policy,
            node_1=node_1,
            node_2=node_2,
            federated_request=FederatedRequest(
                message=message_1, nodes=[node_1, node_2], policy=[policy]
            ),
        )


def test_federated_request_init(federated_request_env):
    env = federated_request_env
    message = MessagesByNode(
        {
            "node-1": env.message_1,
            "node-3": env.message_1,
        },  # prints warning message
    )

    r = FederatedRequest(
        message=message, nodes=[env.node_1, env.node_2], policy=[env.policy]
    )

    assert len(r.requests) == 1


def test_federated_request_send(federated_request_env):
    env = federated_request_env
    env.federated_request.send()

    env.node_1.send.assert_called_once_with(env.message_1, ANY)
    env.node_2.send.assert_called_once_with(env.message_1, ANY)


def test_federated_request_wait(federated_request_env):
    env = federated_request_env
    env.policy_mock.return_value.continue_all.side_effect = [
        PolicyStatus.CONTINUE,
        PolicyStatus.COMPLETED,
    ]
    env.federated_request.wait()
    assert env.policy_mock.return_value.continue_all.call_count == 2


def test_federated_request_with_context_manager(federated_request_env):
    env = federated_request_env
    env.policy_mock.return_value.continue_all.side_effect = [
        PolicyStatus.CONTINUE,
        PolicyStatus.COMPLETED,
    ]

    with FederatedRequest(
        message=env.message_1,
        nodes=[env.node_1, env.node_2],
        policy=[env.policy],
    ) as fed_req:
        assert len(fed_req.requests) == 2
        assert len(fed_req.disconnected_requests()) == 0
        assert fed_req.replies() == {}
        assert fed_req.errors() == {}


# -----------------------------------------------------------------------------
# RequestPolicy
# -----------------------------------------------------------------------------


@pytest.fixture
def policy_env():
    return SimpleNamespace(
        req_1=MagicMock(spec=Request),
        req_2=MagicMock(spec=Request),
        pol=RequestPolicy(),
    )


def test_request_policy_continue(policy_env):
    policy_env.req_1.has_finished.return_value = False
    policy_env.req_2.has_finished.return_value = True

    r = policy_env.pol.continue_([policy_env.req_1, policy_env.req_2])
    assert r == PolicyStatus.CONTINUE


def test_request_policy_stop(policy_env):
    r = policy_env.pol.stop(policy_env.req_1)
    assert r == PolicyStatus.STOPPED
    assert policy_env.pol.status == PolicyStatus.STOPPED
    assert policy_env.pol.stop_caused_by == policy_env.req_1


def test_request_policy_keep(policy_env):
    r = policy_env.pol.keep()
    assert r == PolicyStatus.CONTINUE
    assert policy_env.pol.status == PolicyStatus.CONTINUE


def test_request_policy_completed(policy_env):
    r = policy_env.pol.completed()
    assert r == PolicyStatus.COMPLETED
    assert policy_env.pol.status == PolicyStatus.COMPLETED


# -----------------------------------------------------------------------------
# PolicyController
# -----------------------------------------------------------------------------


@pytest.fixture
def policy_controller_env():
    with patch(
        "fedbiomed.researcher.requests._policies.RequestPolicy"
    ) as req_policy_mock:
        policy_1 = MagicMock()
        yield SimpleNamespace(
            req_policy_mock=req_policy_mock,
            policy_1=policy_1,
            req_1=MagicMock(spec=Request),
            req_2=MagicMock(spec=Request),
            pol_cont=PolicyController(policies=[policy_1]),
        )


def test_policy_controller_continue_all(policy_controller_env):
    env = policy_controller_env
    env.req_policy_mock.return_value.continue_.return_value = PolicyStatus.CONTINUE
    env.policy_1.continue_.return_value = PolicyStatus.CONTINUE

    r = env.pol_cont.continue_all([env.req_1, env.req_1])
    assert r == PolicyStatus.CONTINUE


def test_policy_controller_has_stopped_any(policy_controller_env):
    env = policy_controller_env
    type(env.policy_1).status = PropertyMock(return_value=PolicyStatus.STOPPED)
    assert env.pol_cont.has_stopped_any()


def test_policy_controller_report(policy_controller_env):
    env = policy_controller_env
    type(env.policy_1).status = PropertyMock(return_value=PolicyStatus.STOPPED)
    r = env.pol_cont.report()

    assert r[list(r.keys())[0]]


# -----------------------------------------------------------------------------
# Policy implementations
# -----------------------------------------------------------------------------


@pytest.fixture
def policy_request():
    req = MagicMock(spec=Request)
    node = MagicMock()

    type(node).id = PropertyMock(return_value="node-1")
    type(req).node = PropertyMock(return_value=node)
    type(req).status = PropertyMock()
    type(req).error = PropertyMock()
    return req


def test_discard_on_timeout(policy_request):
    pol = DiscardOnTimeout(nodes=["node-1"], timeout=0.1)

    def time_out():
        time.sleep(1)
        return False

    policy_request.has_finished.side_effect = time_out
    # Unlike StopOnTimeout, a reached timeout discards but never stops
    assert pol.continue_([policy_request]) == PolicyStatus.CONTINUE
    assert pol.continue_([policy_request]) == PolicyStatus.CONTINUE


def test_stop_on_timeout(policy_request):
    pol = StopOnTimeout(nodes=["node-1"], timeout=0.1)

    def time_out():
        time.sleep(1)
        return False

    policy_request.has_finished.side_effect = time_out
    pol.continue_([policy_request])
    r = pol.continue_([policy_request])
    assert r == PolicyStatus.STOPPED


def test_stop_on_disconnect(policy_request):
    pol = StopOnDisconnect(nodes=["node-1"], timeout=0)

    r = pol.continue_([policy_request])
    assert r == PolicyStatus.CONTINUE

    time.sleep(1)
    type(policy_request).status = PropertyMock(return_value=RequestStatus.DISCONNECT)
    r = pol.continue_([policy_request])
    assert r == PolicyStatus.STOPPED


def test_stop_on_error(policy_request):
    pol = StopOnError(nodes=["node-1"])

    type(policy_request).error = PropertyMock(return_value=False)
    r = pol.continue_([policy_request])
    assert r == PolicyStatus.CONTINUE
    type(policy_request).error = PropertyMock(return_value={"error": True})
    r = pol.continue_([policy_request])
    assert r == PolicyStatus.STOPPED


# -----------------------------------------------------------------------------
# Requests under mutual TLS
# -----------------------------------------------------------------------------


@pytest.fixture
def mtls_requests_env():
    """Trust bundle wiring of the researcher server under mutual TLS."""
    with (
        patch(
            "fedbiomed.researcher.requests._requests.GrpcServer", autospec=True
        ) as grpc_server_mock,
        patch(
            "fedbiomed.researcher.requests._requests.is_mtls_enabled",
            return_value=True,
        ),
        tempfile.TemporaryDirectory() as tmp,
    ):
        config_mock = MagicMock()
        config_mock.root = tmp
        db_path = os.path.join(tmp, "certs.json")
        config_mock.getpath.return_value = db_path
        config_mock.config_path = os.path.join(tmp, "config.ini")
        certificate_manager = CertificateManager(db_path=db_path)

        if Requests in Requests._objects:
            del Requests._objects[Requests]

        yield SimpleNamespace(
            grpc_server_mock=grpc_server_mock,
            config=config_mock,
            certificate_manager=certificate_manager,
        )

        certificate_manager.close()
        if Requests in Requests._objects:
            del Requests._objects[Requests]


def test_mtls_without_registered_node_certificate_raises(mtls_requests_env):
    """gRPC cannot bind an empty trust bundle, so it must fail early."""
    with pytest.raises(FedbiomedCertificateError) as exc_info:
        Requests(config=mtls_requests_env.config)

    assert "certificate register" in str(exc_info.value)
    mtls_requests_env.grpc_server_mock.assert_not_called()


def test_mtls_passes_trust_bundle_provider_to_server(mtls_requests_env):
    """The server receives a provider, not a static bundle."""
    mtls_requests_env.certificate_manager.insert(
        certificate="PEM-1",
        party_id="NODE_1",
        component=ComponentType.NODE.name,
    )

    with patch(
        "fedbiomed.researcher.requests._requests.SSLCredentials"
    ) as ssl_credentials:
        Requests(config=mtls_requests_env.config)

    bundle = ssl_credentials.call_args.kwargs["trusted_node_certificates"]
    assert isinstance(bundle, TrustedCertificateBundle)
    assert bundle() == b"PEM-1"
