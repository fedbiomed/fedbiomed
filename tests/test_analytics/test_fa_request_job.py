from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import ErrorNumbers, Stats
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.researcher.federated_workflows.jobs._fa_request_job import FARequestJob
from fedbiomed.researcher.requests import MessagesByNode


@pytest.fixture
def fa_job_setup():
    experiment_id = "exp_id"
    fa_id = "fa_id"
    federated_dataset = MagicMock()
    federated_dataset.data.return_value = {
        "node1": {"dataset_id": "dataset1"},
        "node2": {"dataset_id": "dataset2"},
    }
    stats_args = {"arg1": "value1"}
    stats = [Stats.MEAN.value]
    dataset_schema = ["col1", "col2"]
    nodes = ["node1", "node2"]
    policies = MagicMock()

    reqs = MagicMock()
    researcher_id = "res_id"

    job = FARequestJob(
        experiment_id=experiment_id,
        fa_id=fa_id,
        federated_dataset=federated_dataset,
        stats_args=stats_args,
        stats=stats,
        dataset_schema=dataset_schema,
        nodes=nodes,
        requests=reqs,
        researcher_id=researcher_id,
    )
    job._policies = policies

    return {
        "job": job,
        "experiment_id": experiment_id,
        "fa_id": fa_id,
        "federated_dataset": federated_dataset,
        "stats_args": stats_args,
        "stats": stats,
        "dataset_schema": dataset_schema,
        "nodes": nodes,
        "policies": policies,
        "reqs": reqs,
        "researcher_id": researcher_id,
    }


def test_init(fa_job_setup):
    """Test that all constructor arguments are stored correctly."""
    job = fa_job_setup["job"]
    assert job._experiment_id == fa_job_setup["experiment_id"]
    assert job._fa_id == fa_job_setup["fa_id"]
    assert job._federated_dataset == fa_job_setup["federated_dataset"]
    assert job._stats_args == fa_job_setup["stats_args"]
    assert job._dataset_schema == fa_job_setup["dataset_schema"]
    assert job._stats == fa_job_setup["stats"]
    assert job._researcher_id == fa_job_setup["researcher_id"]


def test_init_raises_when_no_stats_provided(fa_job_setup):
    """Test that __init__ raises when both stats and stats_args are falsy."""
    setup = fa_job_setup
    with pytest.raises(FedbiomedError, match="stats"):
        FARequestJob(
            experiment_id=setup["experiment_id"],
            fa_id=setup["fa_id"],
            federated_dataset=setup["federated_dataset"],
            stats_args=None,
            stats=None,
            dataset_schema=setup["dataset_schema"],
            nodes=setup["nodes"],
            requests=setup["reqs"],
            researcher_id=setup["researcher_id"],
        )


def test_init_with_stats_only(fa_job_setup):
    """Test that __init__ succeeds when only stats is provided (stats_args=None)."""
    setup = fa_job_setup
    job = FARequestJob(
        experiment_id=setup["experiment_id"],
        fa_id=setup["fa_id"],
        federated_dataset=setup["federated_dataset"],
        stats_args=None,
        stats=setup["stats"],
        dataset_schema=setup["dataset_schema"],
        nodes=setup["nodes"],
        requests=setup["reqs"],
        researcher_id=setup["researcher_id"],
    )
    assert job._stats == setup["stats"]
    assert job._stats_args is None


def test_init_with_stats_args_only(fa_job_setup):
    """Test that __init__ succeeds when only stats_args is provided (stats=None)."""
    setup = fa_job_setup
    job = FARequestJob(
        experiment_id=setup["experiment_id"],
        fa_id=setup["fa_id"],
        federated_dataset=setup["federated_dataset"],
        stats_args=setup["stats_args"],
        stats=None,
        dataset_schema=setup["dataset_schema"],
        nodes=setup["nodes"],
        requests=setup["reqs"],
        researcher_id=setup["researcher_id"],
    )
    assert job._stats is None
    assert job._stats_args == setup["stats_args"]


def test_execute_success(fa_job_setup):
    """Test execute sends correct per-node FARequests and returns all replies."""
    job = fa_job_setup["job"]
    reqs = fa_job_setup["reqs"]
    experiment_id = fa_job_setup["experiment_id"]
    fa_id = fa_job_setup["fa_id"]
    stats = fa_job_setup["stats"]
    nodes = fa_job_setup["nodes"]

    # Mock responses
    replies = {
        "node1": FAReply(
            researcher_id="res_id",
            experiment_id=experiment_id,
            fa_id=fa_id,
            node_id="node1",
            node_name="node1_name",
            stats=stats,
            output={"res": 1},
        ),
        "node2": FAReply(
            researcher_id="res_id",
            experiment_id=experiment_id,
            fa_id=fa_id,
            node_id="node2",
            node_name="node2_name",
            stats=stats,
            output={"res": 2},
        ),
    }

    responses_mock = MagicMock()
    responses_mock.errors.return_value = {}
    responses_mock.replies.return_value = replies

    # Context manager mock
    reqs.send.return_value.__enter__.return_value = responses_mock

    result_replies = job.execute()

    # Check that requests were sent
    reqs.send.assert_called_once()
    args, _ = reqs.send.call_args
    sent_requests = args[0]
    sent_nodes = args[1]

    assert isinstance(sent_requests, MessagesByNode)
    assert set(sent_nodes) == set(nodes)

    # Check content of requests
    req_node1 = sent_requests["node1"]
    assert isinstance(req_node1, FARequest)
    assert req_node1.dataset_schema == ["col1", "col2"]
    assert req_node1.dataset_id == "dataset1"
    assert req_node1.stats == stats

    assert result_replies == replies


def test_execute_with_errors_raises(fa_job_setup):
    """Test execute raises FedbiomedError when any node returns an error."""

    job = fa_job_setup["job"]
    reqs = fa_job_setup["reqs"]
    experiment_id = fa_job_setup["experiment_id"]
    fa_id = fa_job_setup["fa_id"]
    stats = fa_job_setup["stats"]

    errors = {
        "node2": ErrorMessage(
            researcher_id="res_id",
            node_id="node2",
            node_name="node2_name",
            errnum=ErrorNumbers.FB325.value,
            extra_msg="Error occurred",
        )
    }
    replies = {
        "node1": FAReply(
            researcher_id="res_id",
            experiment_id=experiment_id,
            fa_id=fa_id,
            node_id="node1",
            node_name="node1_name",
            stats=stats,
            output={"res": 1},
        )
    }

    responses_mock = MagicMock()
    responses_mock.errors.return_value = errors
    responses_mock.replies.return_value = replies

    reqs.send.return_value.__enter__.return_value = responses_mock

    with pytest.raises(FedbiomedError, match="node2"):
        job.execute()


def test_execute_no_replies_raises(fa_job_setup):
    """Test execute raises FedbiomedError when there are no errors but also no replies.

    This covers the case where all nodes disconnected silently.
    """
    job = fa_job_setup["job"]
    reqs = fa_job_setup["reqs"]

    responses_mock = MagicMock()
    responses_mock.errors.return_value = {}
    responses_mock.replies.return_value = {}

    reqs.send.return_value.__enter__.return_value = responses_mock

    with pytest.raises(FedbiomedError, match="No successful replies"):
        job.execute()


def test_execute_errors_are_logged(fa_job_setup):
    """Test that node errors are logged before the exception is raised."""
    job = fa_job_setup["job"]
    reqs = fa_job_setup["reqs"]

    errors = {
        "node1": ErrorMessage(
            researcher_id="res_id",
            node_id="node1",
            node_name="node1_name",
            errnum=ErrorNumbers.FB325.value,
            extra_msg="something went wrong",
        )
    }

    responses_mock = MagicMock()
    responses_mock.errors.return_value = errors
    responses_mock.replies.return_value = {}

    reqs.send.return_value.__enter__.return_value = responses_mock

    with patch(
        "fedbiomed.researcher.federated_workflows.jobs._fa_request_job.logger"
    ) as mock_logger:
        with pytest.raises(FedbiomedError):
            job.execute()

    mock_logger.error.assert_called_once()
    log_args = mock_logger.error.call_args[0][0]
    assert "node1" in log_args
