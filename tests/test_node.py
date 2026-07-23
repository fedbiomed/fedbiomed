import configparser
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from fedbiomed.common.constants import (
    ComponentType,
    ErrorNumbers,
    Stats,
    __messaging_protocol_version__,
)
from fedbiomed.common.exceptions import FedbiomedCertificateError, FedbiomedError
from fedbiomed.common.message import (
    ErrorMessage,
    FARequest,
    ListRequest,
    Message,
    PingReply,
    PingRequest,
    PreprocRequest,
    SearchRequest,
    SecaggDeleteRequest,
    SecaggRequest,
    TrainingPlanStatusRequest,
    TrainRequest,
)
from fedbiomed.node.node import Node, NodeConfig
from fedbiomed.node.round import Round

#############################################################

train_request = TrainRequest(
    researcher_id="researcher-id",
    experiment_id="experiment-1",
    training_plan="x",
    dataset_id="dataset_id_1234",
    training_args={},
    model_args={},
    training=True,
    training_plan_class="MyTrainingPlan",
    aggregator_args={},
    state_id=None,
    params={"x": 1},
    round=1,
)

secagg_request = SecaggRequest(
    researcher_id="r1",
    experiment_id="experiment-id",
    secagg_id="secagg-id",
    parties=["n1", "n2"],
    element=0,
)

ping_request = PingRequest(researcher_id="researcher_id")

list_request = ListRequest(researcher_id="researcher-id")

search_request = SearchRequest(researcher_id="researcher-id", tags=["data"])

tp_status_request = TrainingPlanStatusRequest(
    researcher_id="researcher-id",
    experiment_id="experiment-id",
    training_plan="class MM:;pass",
)

secagg_delete_request = SecaggDeleteRequest(
    researcher_id="researcher_id_1234",
    secagg_id="my_test_secagg_id",
    element=0,
    experiment_id="a_dummy_experiment_id",
)

fa_request = FARequest(
    researcher_id="researcher-id",
    experiment_id="experiment-id",
    dataset_id="dataset-id",
    fa_id="fa-id",
    stats=[Stats.MEAN.value],
    stats_args={},
)

preproc_request = PreprocRequest(
    researcher_id="researcher-id_123",
    experiment_id="experiment-id_456",
    dataset_id="dataset-id_789",
    preproc_id="preproc-id_abc",
    preproc_type=1,
    preproc_step=5,
    preproc_args={"dummy_arg": 42},
    state_id="my_state_id_xyz",
)

database_val = [
    {
        "database_id": "1234",
        "path": "/path/to/my/dataset",
        "name": "test_dataset",
    }
]
database_list = [
    {
        "database_id": "1234",
        "path": "/path/to/my/dataset",
        "name": "test_dataset1",
    },
    {
        "database_id": "5678",
        "path": "/path/to/another/dataset",
        "name": "test_dataset2",
    },
]

database_id = {
    "database_id": "1234",
    "path": "/path/to/my/dataset",
    "name": "test_dataset1",
}


@pytest.fixture
def node_env():
    """Node instance with transport, queue and manager dependencies patched."""
    with (
        patch(
            "fedbiomed.transport.controller.GrpcController.__init__",
            autospec=True,
            return_value=None,
        ),
        patch(
            "fedbiomed.transport.controller.GrpcController.send", autospec=True
        ) as grpc_send,
        patch(
            "fedbiomed.common.tasks_queue.TasksQueue.__init__",
            autospec=True,
            return_value=None,
        ),
        patch("fedbiomed.node.node.EventWaitExchange", autospec=True),
        patch("fedbiomed.node.node.NodeToNodeRouter", autospec=True),
        patch("fedbiomed.node.node.DatasetManager", autospec=True) as dataset_manager,
        patch(
            "fedbiomed.node.node.TrainingPlanSecurityManager", autospec=True
        ) as tp_security_manager,
    ):
        model_manager = MagicMock()
        tp_security_manager.return_value = model_manager

        # mocks
        dataset_manager.return_value.dataset_table.search_by_tags = MagicMock(
            return_value=database_val
        )
        dataset_manager.return_value.list_my_datasets = MagicMock(
            return_value=database_list
        )
        dataset_manager.return_value.reply_training_plan_status_request = MagicMock(
            return_value=None
        )
        dataset_manager.return_value.obfuscate_private_information.side_effect = (
            lambda x: x
        )
        dataset_manager.return_value.get_dataset_entry_by_id = MagicMock(
            return_value=(database_id, "dummy_table_name")
        )

        temp_dir = tempfile.TemporaryDirectory()
        db = os.path.join(temp_dir.name, "test-db.json")
        # creating Node objects
        node_config = NodeConfig(temp_dir.name)
        cfg = configparser.ConfigParser()
        cfg["default"] = {"id": "test-id", "name": "test-name", "db": db}
        cfg["researcher"] = {"ip": "test", "port": "5151"}
        cfg["security"] = {
            "hashing_algorithm": "SHA256",
            "training_plan_approval": "True",
            "allow_preproc": "True",
            "allow_federated_analytics": "True",
            "minimum_samples": "0",
            "secure_aggregation": "False",
            "force_secure_aggregation": "False",
        }
        node_config._cfg = cfg

        yield SimpleNamespace(
            node=Node(node_config),
            node_config=node_config,
            grpc_send=grpc_send,
            model_manager=model_manager,
        )
        temp_dir.cleanup()


@pytest.fixture
def mtls_node_env(node_env):
    """node_env with mutual TLS enabled and a readable node keypair configured."""
    with (
        patch("fedbiomed.node.node.is_mtls_enabled", return_value=True),
        patch(
            "fedbiomed.node.node.read_file", side_effect=["NODE_KEY", "NODE_CERT"]
        ) as read_file,
        patch("fedbiomed.node.node.CertificateManager") as certificate_manager,
    ):
        node_env.node.config._cfg["certificate"] = {
            "private_key": "node.key",
            "public_key": "node.pem",
        }
        node_env.read_file = read_file
        node_env.certificate_manager = certificate_manager
        yield node_env


def test_node_researcher_credentials_mtls(mtls_node_env):
    """Under mutual TLS the node loads its identity and pins the researcher cert."""
    certificate_manager = mtls_node_env.certificate_manager
    certificate_manager.return_value.get_by_component.return_value = ["RES_CERT"]

    credentials = mtls_node_env.node._researcher_credentials()

    assert credentials.mtls
    assert credentials.node_identity.private_key == b"NODE_KEY"
    assert credentials.node_identity.certificate_chain == b"NODE_CERT"
    assert credentials.certificate == b"RES_CERT"
    certificate_manager.return_value.get_by_component.assert_called_once_with(
        ComponentType.RESEARCHER.name
    )
    # The node private key must not leak through the credentials repr.
    assert "NODE_KEY" not in repr(credentials)


def test_node_researcher_credentials_mtls_missing_researcher_cert(mtls_node_env):
    """mTLS enabled but no registered researcher certificate is a hard error."""
    mtls_node_env.certificate_manager.return_value.get_by_component.return_value = []

    with pytest.raises(FedbiomedCertificateError):
        mtls_node_env.node._researcher_credentials()


def test_node_researcher_credentials_mtls_ambiguous_researcher_cert(mtls_node_env):
    """Several registered researcher certificates make the one to pin ambiguous."""
    mtls_node_env.certificate_manager.return_value.get_by_component.return_value = [
        "RES_CERT_1",
        "RES_CERT_2",
    ]

    with pytest.raises(FedbiomedCertificateError) as exc_info:
        mtls_node_env.node._researcher_credentials()

    assert "ambiguous" in str(exc_info.value)


def test_node_researcher_credentials_mtls_unreadable_node_cert(mtls_node_env):
    """A missing/unreadable node key or cert surfaces as FedbiomedCertificateError."""
    mtls_node_env.read_file.side_effect = FedbiomedError("cannot read file")

    with pytest.raises(FedbiomedCertificateError):
        mtls_node_env.node._researcher_credentials()


@patch("fedbiomed.common.tasks_queue.TasksQueue.add")
def test_node_add_task_normal_case_scenario(task_queue_add, node_env):
    """Tests add_task method (in the normal case scenario)"""

    node_env.node.add_task(train_request)
    task_queue_add.assert_called_once_with(train_request)


@pytest.mark.parametrize("request_", [train_request, secagg_request])
@patch("fedbiomed.common.tasks_queue.TasksQueue.add")
def test_node_on_message_normal_case_scenario_train_secagg_reply(
    task_queue_add, node_env, request_
):
    """Tests `on_message` method (normal case scenario), with train/secagg command"""
    message = request_.to_dict()

    # action
    node_env.node.on_message(message)

    # checks
    task_queue_add.assert_called_once_with(Message.from_dict(message))


def test_node_on_message_normal_case_scenario_ping(node_env):
    """Tests `on_message` method (normal case scenario), with ping command"""

    # action
    node_env.node.on_message(ping_request.to_dict())
    node_env.grpc_send.assert_called_once()


@patch("fedbiomed.common.tasks_queue.TasksQueue.add")
def test_node_on_message_train_logs_request_lifecycle(task_queue_add, node_env):
    """Tests that train requests emit the new structured debug logs."""

    with patch("fedbiomed.node.node.logger.debug") as logger_debug:
        node_env.node.on_message(train_request.to_dict())

    task_queue_add.assert_called_once()

    debug_calls = logger_debug.call_args_list
    assert any(
        call.args[0]
        == "Received researcher message type=%s req=%s researcher=%s experiment=%s dataset=%s round=%s"
        and call.args[1:]
        == (
            train_request.__name__,
            getattr(train_request, "request_id", None),
            train_request.researcher_id,
            train_request.experiment_id,
            train_request.dataset_id,
            train_request.round,
        )
        for call in debug_calls
    )
    assert any(
        call.args[0] == "Queueing node task type=%s req=%s experiment=%s"
        and call.args[1:]
        == (
            train_request.__name__,
            getattr(train_request, "request_id", None),
            train_request.experiment_id,
        )
        for call in debug_calls
    )


@patch("fedbiomed.node.node.SecaggManager")
def test_node_on_message_normal_case_scenario_secagg_delete(skm, node_env):
    """Tests `on_message` method (normal case scenario), with secagg-delete command"""

    skm.return_value.return_value.remove.return_value = True
    node_env.node.on_message(secagg_delete_request.to_dict())
    node_env.grpc_send.assert_called_once()


def test_node_on_message_normal_case_scenario_search(node_env):
    """Tests `on_message` method (normal case scenario), with search command"""
    # action
    node_env.node.on_message(search_request.to_dict())
    node_env.grpc_send.assert_called_once()


def test_node_on_message_normal_case_scenario_list(node_env):
    """Tests `on_message` method (normal case scenario), with list command"""

    # action
    node_env.node.on_message(list_request.to_dict())
    node_env.grpc_send.assert_called_once()


def test_node_on_message_normal_case_scenario_model_status(node_env):
    """Tests normal case scenario, if command is equals to 'training-plan-status"""

    node_env.node.on_message(tp_status_request.to_dict())
    node_env.model_manager.reply_training_plan_status_request.assert_called_once_with(
        tp_status_request
    )


def test_node_on_message_unknown_command(node_env):
    """Tests Exception is handled if command is not a known command
    (in `on_message` method)"""
    ping_reply = PingReply(researcher_id="r1", node_id="n1", node_name="n1_name")

    # action
    node_env.node.on_message(ping_reply.to_dict())
    error = node_env.grpc_send.call_args.args[1]
    assert isinstance(error, ErrorMessage)


def test_node_on_message_fail_msg_not_deserializable(node_env):
    """Tests case where a error raised (because unable to deserialize message)"""
    # Not desearializable
    ping_msg = {"researcher_id": "re1", "request_id": "1234"}

    node_env.node.on_message(ping_msg)

    error = node_env.grpc_send.call_args.args[1]
    assert isinstance(error, ErrorMessage)


@patch("fedbiomed.node.node.Round", autospec=True)
@patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__", spec=True)
def test_node_parser_task_train_create_round(
    history_monitor_patch, round_patch, node_env
):
    """Tests if rounds are created accordingly - running normal case scenario
    (in `parser_task_train` method)"""

    history_monitor_patch.return_value = None
    round_patch.return_value.initialize_arguments.return_value = None

    round_ = node_env.node.parser_task_train(train_request)
    assert isinstance(round_, Round)
    round_patch.assert_called_once()


@patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__")
@patch("fedbiomed.node.round.Round.__init__")
def test_node_parser_task_train_no_dataset_found(
    round_init, history_monitor_patch, node_env
):
    """Tests parser_task_train method, case where no dataset has been found"""
    # defining patchers
    history_monitor_patch.return_value = None
    round_init.return_value = None

    mock_dataset_manager = MagicMock()
    mock_dataset_manager.get_dataset_entry_by_id = MagicMock(return_value=(None, None))
    node_env.node.dataset_manager = mock_dataset_manager
    with patch("fedbiomed.node.node.logger.error") as logger_error:
        node_env.node.parser_task_train(train_request)

        assert logger_error.call_count >= 1
        messages = [call.args[0] for call in logger_error.call_args_list]
        assert any(ErrorNumbers.FB313.value in m for m in messages)

    error = node_env.grpc_send.call_args.args[1]
    assert isinstance(error, ErrorMessage)


@patch("fedbiomed.node.node.Round", autospec=True)
@patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__", spec=True)
def test_node_parser_task_train_initialize_arguments_failure_returns_none_and_sends_error(
    history_monitor_patch, round_patch, node_env
):
    """If Round.initialize_arguments raises, parser_task_train must send an error and return None."""

    history_monitor_patch.return_value = None
    round_patch.return_value.initialize_arguments.side_effect = Exception(
        "init-args-boom"
    )

    with (
        patch("fedbiomed.node.node.logger.error") as logger_error,
        patch("fedbiomed.node.node.logger.debug") as logger_debug,
    ):
        round_ = node_env.node.parser_task_train(train_request)

    assert round_ is None
    assert logger_error.called
    assert logger_debug.called

    error = node_env.grpc_send.call_args.args[1]
    assert isinstance(error, ErrorMessage)


@pytest.mark.parametrize(
    "secagg_arguments,round_",
    [
        (
            {
                "secagg_servkey_id": None,
                "secagg_random": None,
                "secagg_clipping_range": None,
            },
            1,
        ),
        (None, 0),
    ],
)
@patch("fedbiomed.node.node.Round", autospec=True)
@patch("fedbiomed.node.history_monitor.HistoryMonitor.__init__", spec=True)
def test_node_parser_task_train_maps_request_to_round_arguments(
    history_monitor_patch, round_patch, node_env, secagg_arguments, round_
):
    """A train request is mapped field by field onto the Round constructor,
    with and without secagg arguments."""

    request_dict = {
        "protocol_version": str(__messaging_protocol_version__),
        "model_args": {"lr": 0.1},
        "training_args": {"some_value": 1234},
        "training_plan": "TP",
        "training_plan_class": "my_test_training_plan",
        "params": {"x": 0},
        "experiment_id": "experiment_id_1234",
        "state_id": None,
        "secagg_arguments": secagg_arguments,
        "round": round_,
        "researcher_id": "researcher_id_1234",
        "dataset_id": "dataset_id_1234",
        "training": True,
        "aggregator_args": {},
        "optim_aux_var": None,
    }
    history_monitor_patch.return_value = None
    round_patch.return_value.initialize_arguments.return_value = None

    node_env.node.parser_task_train(TrainRequest(**request_dict))

    round_patch.assert_called_once_with(
        root_dir=node_env.node_config.root,
        db=node_env.node_config.get("default", "db"),
        node_id=node_env.node_config.get("default", "id"),
        node_name=node_env.node_config.get("default", "name"),
        training_plan=request_dict["training_plan"],
        training_plan_class=request_dict["training_plan_class"],
        model_kwargs=request_dict["model_args"],
        training_kwargs=request_dict["training_args"],
        training=True,
        dataset_entry=database_id,
        params=request_dict["params"],
        experiment_id=request_dict["experiment_id"],
        researcher_id=request_dict["researcher_id"],
        history_monitor=ANY,
        aggregator_args=None,
        node_args={},
        tp_security_manager=ANY,
        round_number=round_,
        dlp_and_loading_block_metadata=None,
        aux_vars=request_dict["optim_aux_var"],
    )


@patch("fedbiomed.common.tasks_queue.TasksQueue.get")
def test_node_task_manager_exception_raised_task_queue(tasks_queue_get, node_env):
    """Simulates an Exception (SystemError) triggered by `tasks_queue.get`"""
    tasks_queue_get.side_effect = SystemError("mimicking an exception")

    with pytest.raises(SystemError):
        node_env.node.task_manager()


@patch("fedbiomed.common.tasks_queue.TasksQueue.task_done")
@patch("fedbiomed.node.node.Node._task_secagg")
@patch("fedbiomed.common.tasks_queue.TasksQueue.get")
def test_node_task_manager_secagg_exception_raised_task_done(
    tasks_queue_get, task_secagg, tasks_queue_task_done, node_env
):
    """Tests if an Exception (SystemExit) is triggered when calling
    `TasksQueue.task_done` method for secagg message"""
    tasks_queue_get.return_value = {
        "protocol_version": "99.99",
        "researcher_id": "my_test_researcher",
        "secagg_id": "my_test_secagg",
        "element": 33,
        "experiment_id": "my_experiment",
        "parties": [],
    }
    task_secagg.return_value = None
    tasks_queue_task_done.side_effect = SystemExit(
        "Mimicking an exception happening in `TasksQueue.task_done` method"
    )

    with pytest.raises(SystemExit):
        node_env.node.task_manager()

    # No reply is sent when task_done aborts the loop
    assert node_env.grpc_send.call_count == 0


@patch("fedbiomed.transport.controller.GrpcController.start")
def test_node_start_messaging_normal_case_scenario(msg_start, node_env):
    """Tests `start_messaging` method (normal_case_scenario)"""
    node_env.node.start_messaging(True)
    msg_start.assert_called_once_with(True)


def test_node_send_error_logs_debug_context(node_env):
    """Tests `send_error` emits the new debug traces before and after dispatch."""

    errnum = ErrorNumbers.FB100
    extra_msg = "this is a test_send_error"
    researcher_id = "researcher_id_1224"

    with (
        patch("fedbiomed.node.node.logger.debug") as logger_debug,
        patch("fedbiomed.node.node.logger.error") as logger_error,
    ):
        node_env.node.send_error(errnum, extra_msg, researcher_id)

    node_env.grpc_send.assert_called_once()
    logger_error.assert_called_once()

    debug_calls = logger_debug.call_args_list
    assert any(
        call.args[0]
        == "Preparing error reply errnum=%s req=%s researcher=%s broadcast=%s connected=%s destination=%s:%s msg_len=%d"
        and call.args[1:]
        == (
            errnum.name,
            None,
            researcher_id,
            False,
            False,
            "test",
            "5151",
            len(extra_msg),
        )
        and call.kwargs.get("stack_info") is True
        for call in debug_calls
    )
    assert any(
        call.args[0]
        == "Error reply dispatched errnum=%s req=%s researcher=%s broadcast=%s connected=%s"
        and call.args[1:] == (errnum.name, None, researcher_id, False, False)
        for call in debug_calls
    )


@patch("fedbiomed.node.node.SecaggSetup")
def test_node_task_secagg(secagg_setup, node_env):
    """Tests `_task_secagg` normal (successful) case"""
    # Test .setup()execution. It is normal the get result as success False since setup will fail
    # due to not existing certificate files
    x = MagicMock()
    secagg_setup.return_value.return_value = x
    x.setup.return_value = MagicMock()
    node_env.node._task_secagg(secagg_request)

    # Test setup error case ---------------------------------------------------------------
    x.setup.side_effect = Exception
    node_env.node._task_secagg(secagg_request)
    error = node_env.grpc_send.call_args.args[1]
    assert isinstance(error, ErrorMessage)


def test_node_task_secagg_delete(node_env):
    """Tests `_task_secagg_delete` failure replies"""

    request = SecaggDeleteRequest(
        protocol_version=str(__messaging_protocol_version__),
        researcher_id="party1",
        secagg_id="my_dummy_secagg_id",
        request_id="request",
        element=0,
        experiment_id="my_test_experiment",
    )

    # Remove fails since there is no registry in DB
    node_env.node._task_secagg_delete(request)
    error = node_env.grpc_send.call_args.args[1]
    assert isinstance(error, ErrorMessage)
    node_env.grpc_send.reset_mock()

    # Remove reports failure
    with patch("fedbiomed.node.node.SecaggManager") as skm:
        skm.return_value.return_value.remove.return_value = False
        node_env.node._task_secagg_delete(request)
        error = node_env.grpc_send.call_args.args[1]
        assert isinstance(error, ErrorMessage)


def test_node_on_message_fa_request(node_env):
    """Tests `on_message` method with FARequest"""
    with patch.object(node_env.node, "add_task") as mock_add_task:
        node_env.node.on_message(fa_request.to_dict())
        mock_add_task.assert_called_once()
        args, _ = mock_add_task.call_args
        assert isinstance(args[0], FARequest)


@patch("fedbiomed.common.tasks_queue.TasksQueue.get")
@patch("fedbiomed.common.tasks_queue.TasksQueue.task_done")
@patch("fedbiomed.node.node.FAJob")
def test_node_task_manager_fa_request(mock_fa_job, mock_task_done, mock_get, node_env):
    """Tests `task_manager` with FARequest"""
    mock_get.side_effect = [fa_request, SystemExit]

    # Mock FAJob run method
    mock_job_instance = mock_fa_job.return_value
    mock_job_instance.run.return_value = MagicMock()

    with pytest.raises(SystemExit):
        node_env.node.task_manager()

    mock_job_instance.run.assert_called_once()
    node_env.grpc_send.assert_called_once()


@patch("fedbiomed.common.tasks_queue.TasksQueue.get")
@patch("fedbiomed.common.tasks_queue.TasksQueue.task_done")
@patch("fedbiomed.node.node.PreprocJob")
def test_node_task_manager_preproc_request(
    mock_preproc_job, mock_task_done, mock_get, node_env
):
    """Tests `task_manager` with PreprocRequest"""
    mock_get.side_effect = [preproc_request, SystemExit]

    # Mock PreprocJob run method
    mock_job_instance = mock_preproc_job.return_value
    mock_job_instance.run.return_value = MagicMock()

    with pytest.raises(SystemExit):
        node_env.node.task_manager()

    mock_job_instance.run.assert_called_once()
    node_env.grpc_send.assert_called_once()
