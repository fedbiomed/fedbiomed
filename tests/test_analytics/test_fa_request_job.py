from unittest.mock import MagicMock

import pytest

from fedbiomed.common.constants import ErrorNumbers, Stats
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
    fa_args = {"arg1": "value1"}
    stats = Stats.MEAN.value
    dataset_args = {"d_arg1": "value1"}
    dataset_schema = ["col1", "col2"]
    nodes = ["node1", "node2"]
    policies = MagicMock()

    reqs = MagicMock()
    researcher_id = "res_id"

    job = FARequestJob(
        experiment_id=experiment_id,
        fa_id=fa_id,
        federated_dataset=federated_dataset,
        fa_args=fa_args,
        stats=stats,
        dataset_args=dataset_args,
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
        "fa_args": fa_args,
        "stats": stats,
        "dataset_args": dataset_args,
        "dataset_schema": dataset_schema,
        "nodes": nodes,
        "policies": policies,
        "reqs": reqs,
        "researcher_id": researcher_id,
    }


def test_init(fa_job_setup):
    """Test initialization of FARequestJob"""
    job = fa_job_setup["job"]
    assert job._experiment_id == fa_job_setup["experiment_id"]
    assert job._fa_id == fa_job_setup["fa_id"]
    assert job._federated_dataset == fa_job_setup["federated_dataset"]
    assert job._fa_args == fa_job_setup["fa_args"]
    assert job._dataset_args == fa_job_setup["dataset_args"]
    assert job._dataset_schema == fa_job_setup["dataset_schema"]
    assert job._stats == fa_job_setup["stats"]
    assert job._researcher_id == fa_job_setup["researcher_id"]


def test_execute_success(fa_job_setup):
    """Test execute method with successful replies"""
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

    result_replies, result_errors = job.execute()

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
    assert result_errors == {}


def test_execute_with_errors(fa_job_setup):
    """Test execute method when some nodes return errors"""
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

    result_replies, result_errors = job.execute()

    assert result_replies == replies
    assert result_errors == errors
