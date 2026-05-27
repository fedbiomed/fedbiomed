from unittest.mock import patch

import pytest

from fedbiomed.common.constants import ErrorNumbers, PreprocType
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
        yield mock_dm


@pytest.fixture
def preproc_job_args(preproc_request, mocked_dataset_manager):
    return {
        "root_dir": "/tmp",
        "dataset_manager": mocked_dataset_manager,
        "node_id": "node_1",
        "node_name": "Toto",
        "request": preproc_request,
        "allow_preproc": True,
    }


@pytest.fixture(autouse=True)
def preproc_type_to_jobs(monkeypatch):
    """Ensure tests see a mutable _preproc_type_to_jobs mapping they can populate."""

    class DummyJob:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return {"result": "success"}

    preproc_type = PreprocType.FEDCOMBAT
    monkeypatch.setattr(_preproc_job, "_preproc_type_to_jobs", {preproc_type: DummyJob})
    return {preproc_type: DummyJob}


@pytest.fixture
def preproc_type(monkeypatch):
    """Ensure PreprocType is constructible in tests."""
    preproc_type = PreprocType.FEDCOMBAT
    monkeypatch.setattr(_preproc_job, "PreprocType", lambda x: preproc_type)

    return preproc_type


@pytest.fixture(autouse=True)
def preproc_type_to_steps(monkeypatch):
    """Ensure tests see a mutable _preproc_step_to_jobs mapping they can populate."""

    class DummyStepJob:
        def __init__(self, v, *args, **kwargs):
            self.name = v

        def __call__(self, *args, **kwargs):
            return {"result": "success"}

    preproc_type = PreprocType.FEDCOMBAT
    monkeypatch.setattr(
        _preproc_job, "_preproc_type_to_steps", {preproc_type: DummyStepJob}
    )
    return {preproc_type: DummyStepJob}


@pytest.fixture
def preproc_step(monkeypatch):
    """Ensure HarmonizationStep is constructible in tests."""

    monkeypatch.setattr(_preproc_job, "HarmonizationStep", "DummyStepJob")

    return "DummyStepJob"


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


def test_run_success(
    monkeypatch,
    preproc_request,
    preproc_job_args,
    preproc_type,
    preproc_step,
):
    """Test successful run of PreprocJob."""

    job = PreprocJob(**preproc_job_args)
    reply = job.run()

    # intermediate value settings checks
    assert job._preproc_step.name == preproc_request.preproc_step
    assert job._preproc_type.name == preproc_type.name
    # final reply checks
    assert isinstance(reply, PreprocReply)
    assert reply.request_id == preproc_request.request_id
    assert reply.researcher_id == preproc_request.researcher_id
    assert reply.experiment_id == preproc_request.experiment_id
    assert reply.node_id == preproc_job_args["node_id"]
    assert reply.node_name == preproc_job_args["node_name"]
    assert reply.state_id == preproc_request.state_id


def test_run_dataset_entry_not_dict(
    monkeypatch,
    preproc_job_args,
    mocked_dataset_manager,
    preproc_type,
    preproc_step,
):
    """Test run of PreprocJob when dataset entry returned by DatasetManager is not a dict."""

    # make dataset_table return a non-dict value
    mocked_dataset_manager.dataset_table.get_by_id.return_value = "this_is_not_a_dict"

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "dataset" in (result.extra_msg or "").lower()


def test_run_preproc_not_allowed(preproc_job_args):
    """Test run of PreprocJob when preprocessing is not allowed."""

    job_args = preproc_job_args.copy()
    job_args["allow_preproc"] = False

    job = PreprocJob(**job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "not allowed" in result.extra_msg


def test_run_invalid_preproc_type(monkeypatch, preproc_job_args, preproc_step):
    """Test run of PreprocJob with invalid preproc_type."""

    # Make PreprocType raise ValueError to simulate invalid type
    def bad_type(x):
        raise ValueError("bad")

    monkeypatch.setattr(_preproc_job, "PreprocType", bad_type)

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "invalid preproc_type" in result.extra_msg


def test_run_invalid_preproc_step(monkeypatch, preproc_job_args, preproc_type):
    """Test run of PreprocJob with invalid preproc_step."""

    # Keep a supported PreprocType, but make step constructor fail

    def bad_step(x):
        raise ValueError("bad step")

    monkeypatch.setattr(_preproc_job, "PreprocType", lambda x: preproc_type)
    monkeypatch.setattr(
        _preproc_job, "_preproc_type_to_steps", {preproc_type: bad_step}
    )

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "invalid preproc_step" in result.extra_msg


def test_run_preprocreply_construction_failure(
    monkeypatch,
    preproc_job_args,
    preproc_type,
    preproc_step,
):
    """Test run of PreprocJob where PreprocReply construction fails."""

    class DummyJob:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return {"result": "success"}

    monkeypatch.setattr(_preproc_job, "_preproc_type_to_jobs", {preproc_type: DummyJob})

    # Simulate PreprocReply raising during construction
    def bad_reply(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_preproc_job, "PreprocReply", bad_reply)

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
    assert "Preprocessing job cannot reply" in result.extra_msg


def test_run_no_job_for_type(monkeypatch, preproc_job_args, preproc_type, preproc_step):
    """When no job is registered for the PreprocType, run() should return an ErrorMessage."""

    # remove any registered jobs
    monkeypatch.setattr(_preproc_job, "_preproc_type_to_jobs", {})

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value


def test_run_job_raises_exception(
    monkeypatch, preproc_job_args, preproc_type, preproc_step
):
    """If the registered preproc job raises, run() should return an ErrorMessage."""

    class BadJob:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise RuntimeError("job failed")

    monkeypatch.setattr(_preproc_job, "_preproc_type_to_jobs", {preproc_type: BadJob})

    job = PreprocJob(**preproc_job_args)
    result = job.run()

    assert isinstance(result, ErrorMessage)
    assert result.errnum == ErrorNumbers.FB326.value
