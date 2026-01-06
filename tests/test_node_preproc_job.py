from unittest.mock import patch

import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.message import ErrorMessage, PreprocReply, PreprocRequest
from fedbiomed.node.jobs import _preproc_job
from fedbiomed.node.jobs._preproc_job import PreprocJob


@pytest.fixture
def request_args():
    return {
        "researcher_id": "researcher_1",
        "request_id": "req_1",
        "dataset_id": "dataset_123",
        "experiment_id": "exp_1",
        "preproc_type": 123,
        "preproc_step": 456,
        "preproc_id": "preproc_1",
        "preproc_args": {"arg": 1},
        "state_id": "state_1",
    }


@pytest.fixture
def preproc_request(request_args):
    return PreprocRequest(**request_args)


@pytest.fixture
def mocked_dataset_manager():
    """Fixture for mocked DatasetManager with common setup."""
    with patch("fedbiomed.node.jobs._base_job.DatasetManager") as mock_dm_cls:
        mock_dm = mock_dm_cls.return_value
        mock_dm.dataset_table.get_by_id.return_value = {
            "data_type": "csv",
            "path": "/path/to/data",
            "dataset_parameters": {},
        }
        yield mock_dm_cls, mock_dm


@pytest.fixture
def preproc_job_args(preproc_request):
    return {
        "root_dir": "/tmp",
        "dataset_manager": mocked_dataset_manager,
        "node_id": "node_1",
        "node_name": "Toto",
        "request": preproc_request,
    }


def test_preproc_job_init(preproc_request, preproc_job_args):
    """Test PreprocJob initialization."""
    job = PreprocJob(**preproc_job_args)

    assert job._dir == preproc_job_args["root_dir"]
    assert job._dataset_manager is preproc_job_args["dataset_manager"]
    assert job._node_id == preproc_job_args["node_id"]
    assert job._node_name == preproc_job_args["node_name"]
    assert job._request_id == preproc_request.request_id
    assert job._researcher_id == preproc_request.researcher_id
    assert job._experiment_id == preproc_request.experiment_id
    assert job._preproc_type_raw == preproc_request.preproc_type
    assert job._preproc_step_raw == preproc_request.preproc_step
    assert job._preproc_id == preproc_request.preproc_id
    assert job._preproc_args_raw == preproc_request.preproc_args
    assert job._state_id == preproc_request.state_id


def test_build_args_for_dataset(preproc_job_args):
    # Dummy test to cover _build_args_for_dataset, nothing to check
    job = PreprocJob(**preproc_job_args)
    job._build_args_for_dataset(None)


def test_run_success(monkeypatch, preproc_request, preproc_job_args):
    """Test successful run of PreprocJob."""

    # Patch PreprocType and PreprocStep to simple callables returning objects with .name
    class Dummy:
        def __init__(self, v):
            self.name = v

    monkeypatch.setattr(_preproc_job, "PreprocType", Dummy)
    monkeypatch.setattr(_preproc_job, "PreprocStep", Dummy)

    job = PreprocJob(**preproc_job_args)
    reply = job.run()

    # intermediate value settings checks
    assert job._preproc_step.name == preproc_request.preproc_step
    assert job._preproc_type.name == preproc_request.preproc_type
    # final reply checks
    assert isinstance(reply, PreprocReply)
    assert reply.request_id == preproc_request.request_id
    assert reply.researcher_id == preproc_request.researcher_id
    assert reply.experiment_id == preproc_request.experiment_id
    assert reply.node_id == preproc_job_args["node_id"]
    assert reply.node_name == preproc_job_args["node_name"]
    assert reply.state_id == preproc_request.state_id


def test_run_invalid_preproc_type(monkeypatch, preproc_job_args):
    """Test run of PreprocJob with invalid preproc_type."""

    # Make PreprocType raise ValueError to simulate invalid type
    def bad_type(x):
        raise ValueError("bad")

    class Dummy:
        def __init__(self, v):
            self.name = v

    monkeypatch.setattr(_preproc_job, "PreprocType", bad_type)
    monkeypatch.setattr(_preproc_job, "PreprocStep", Dummy)

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "invalid preproc_type" in result.extra_msg


def test_run_invalid_preproc_step(monkeypatch, preproc_job_args):
    """Test run of PreprocJob with invalid preproc_step."""

    # PreprocType ok, PreprocStep bad
    class DummyType:
        def __init__(self, v):
            self.name = v

    def bad_step(x):
        raise ValueError("bad step")

    monkeypatch.setattr(_preproc_job, "PreprocType", DummyType)
    monkeypatch.setattr(_preproc_job, "PreprocStep", bad_step)

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "invalid preproc_step" in result.extra_msg


def test_run_preprocreply_construction_failure(
    monkeypatch,
    preproc_job_args,
):
    """Test run of PreprocJob where PreprocReply construction fails."""

    # Simulate PreprocReply raising during construction
    class Dummy:
        def __init__(self, v):
            self.name = v

    monkeypatch.setattr(_preproc_job, "PreprocType", Dummy)
    monkeypatch.setattr(_preproc_job, "PreprocStep", Dummy)

    def bad_reply(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_preproc_job, "PreprocReply", bad_reply)

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "Preprocessing job failed" in result.extra_msg
