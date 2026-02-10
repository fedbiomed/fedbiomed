from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import ErrorNumbers, Stats
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.node.jobs._fa_job import FAJob, _InternalJobError


@pytest.fixture
def request_args():
    return {
        "researcher_id": "res_1",
        "request_id": "req_1",
        "experiment_id": "exp_1",
        "dataset_id": "dataset_123",
        "fa_id": "fa_1",
        "stats": Stats.MEAN.value,
        "fa_args": {},
        "dataset_args": {"input_columns": ["age", "weight"]},
        "dataset_schema": ["col1", "col2"],
    }


@pytest.fixture
def fa_request(request_args):
    return FARequest(**request_args)


@pytest.fixture
def fa_job_args(fa_request):
    return {
        "root_dir": "/tmp/root",
        "dataset_manager": MagicMock(),
        "node_id": "node_1",
        "node_name": "test_node",
        "request": fa_request,
        "allow_fa": True,
    }


@pytest.fixture
def fa_job(fa_job_args):
    return FAJob(**fa_job_args)


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
def mocked_dlp():
    """Fixture for mocked DataLoadingPlan."""
    with patch("fedbiomed.node.jobs._base_job.DataLoadingPlan") as mock_dlp_cls:
        mock_dlp = mock_dlp_cls.return_value
        mock_dlp.deserialize.return_value = mock_dlp
        yield mock_dlp_cls, mock_dlp


def test_fa_job_init(fa_job, fa_job_args, request_args):
    """Test FAJob initialization."""
    assert fa_job._dir == fa_job_args["root_dir"]
    assert fa_job._dataset_manager == fa_job_args["dataset_manager"]
    assert fa_job._node_id == fa_job_args["node_id"]
    assert fa_job._node_name == fa_job_args["node_name"]
    assert fa_job._dataset_id == request_args["dataset_id"]
    assert fa_job._experiment_id == request_args["experiment_id"]
    assert fa_job._fa_id == request_args["fa_id"]
    assert fa_job._researcher_id == request_args["researcher_id"]
    assert fa_job._request_id == request_args["request_id"]
    assert fa_job._fa_args == request_args["fa_args"]


def test_fa_job_init_defaults(fa_job_args, request_args):
    """Test FAJob initialization with default arguments."""
    # Create a request with empty fa_args
    req_args = request_args.copy()
    req_args["fa_args"] = {}
    request = FARequest(**req_args)

    args = fa_job_args.copy()
    args["request"] = request

    job = FAJob(**args)
    assert job._fa_args == {}


def test_build_error_msg(fa_job):
    """Test _build_error_msg method."""
    msg = "Something went wrong"
    errnum = ErrorNumbers.FB313.value
    error = fa_job._build_error_msg(msg, errnum)

    assert isinstance(error, ErrorMessage)
    assert error.extra_msg == msg
    assert error.errnum == errnum
    assert error.node_id == fa_job._node_id
    assert error.request_id == fa_job._request_id


@patch("fedbiomed.node.jobs._base_job.REGISTRY_CONTROLLERS")
def test_build_dataset_success(mock_registry, fa_job, mocked_dataset_manager):
    """Test _build_dataset success scenario."""
    mock_dm_cls, mock_dm = mocked_dataset_manager
    mock_dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "path": "/path/to/data",
        "dataset_parameters": {},
    }
    fa_job._dataset_manager = mock_dm

    # When instantiated
    mock_dataset_instance = MagicMock()

    class MockDataset:
        def __init__(self, input_columns=None):
            pass

        def __new__(cls, *args, **kwargs):
            return mock_dataset_instance

    mock_dataset_cls = MockDataset

    # Setup DATASET_CLASSES_PER_TYPE
    mock_registry.__contains__.return_value = True
    # Return 3 values as expected by unpacking: _, _, dataset_cls
    mock_registry.__getitem__.return_value = (None, None, mock_dataset_cls)

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN, ["csv"])

    # Verify dataset creation and initialization
    assert dataset == mock_dataset_instance
    mock_dataset_instance.complete_initialization.assert_called_once()
    call_args = mock_dataset_instance.complete_initialization.call_args
    assert call_args[0][0]["root"] == "/path/to/data"
    assert call_args[0][1] == DataReturnFormat.SKLEARN


def test_build_dataset_not_found(mocked_dataset_manager, fa_job):
    """Test _build_dataset when dataset is not found in DB."""
    mock_dm_cls, mock_dm = mocked_dataset_manager
    mock_dm.dataset_table.get_by_id.return_value = None
    fa_job._dataset_manager = mock_dm

    with pytest.raises(_InternalJobError) as exc_info:
        fa_job._build_dataset(DataReturnFormat.SKLEARN, ["csv"])

    assert "Cannot found request dataset in local datasets" in str(exc_info.value)


@patch("fedbiomed.node.jobs._base_job.REGISTRY_CONTROLLERS", {})
def test_build_dataset_invalid_type(mocked_dataset_manager, fa_job):
    """Test _build_dataset when data type is not supported."""
    mock_dm_cls, mock_dm = mocked_dataset_manager
    mock_dm.dataset_table.get_by_id.return_value = {"data_type": "unknown_type"}

    with pytest.raises(_InternalJobError) as exc_info:
        fa_job._build_dataset(DataReturnFormat.SKLEARN, ["not_a_dataset_supported"])

    assert "not supported" in str(exc_info.value)


@patch("fedbiomed.node.jobs._base_job.REGISTRY_CONTROLLERS")
@patch("fedbiomed.node.jobs._base_job.DataLoadingPlan")
def test_build_dataset_with_dlp_success(
    mock_dlp_cls, mock_reg_cont, fa_job, mocked_dataset_manager
):
    """Test _build_dataset with DataLoadingPlan."""
    mock_dm_cls, mock_dm = mocked_dataset_manager
    mock_dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "path": "/path",
        "dlp_id": "dlp_1",
    }
    mock_dm.get_dlp_by_id.return_value = ["dlp_content"]
    fa_job._dataset_manager = mock_dm

    mock_dlp = mock_dlp_cls.return_value
    mock_dlp.deserialize.return_value = mock_dlp  # return self

    mock_dataset_instance = MagicMock()

    class MockDataset:
        def __init__(self, input_columns=None):
            pass

        def __new__(cls, *args, **kwargs):
            return mock_dataset_instance

    mock_reg_cont.__getitem__.return_value = (None, None, MockDataset)
    mock_reg_cont.__contains__.return_value = True

    # Setup DATASET_CLASSES_PER_TYPE
    # mock_dataset_cli not needed since _BaseJob uses REGISTRY_CONTROLLERS

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN, ["csv"])

    assert dataset == mock_dataset_instance
    # Check if DLP was passed to complete_initialization
    call_args = mock_dataset_instance.complete_initialization.call_args
    assert call_args[0][0]["dlp"] == mock_dlp


@patch("fedbiomed.node.jobs._base_job.REGISTRY_CONTROLLERS")
@patch("fedbiomed.node.jobs._base_job.DataLoadingPlan")
def test_build_dataset_dlp_error(
    mock_dlp_cls, mock_registry, fa_job, mocked_dataset_manager
):
    """Test _build_dataset when DLP deserialization fails."""
    mock_dm_cls, mock_dm = mocked_dataset_manager
    mock_dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "dlp_id": "dlp_1",
    }
    mock_dm.get_dlp_by_id.return_value = ["dlp_content"]
    fa_job._dataset_manager = mock_dm

    mock_dlp_cls.return_value.deserialize.side_effect = FedbiomedError("DLP Error")
    mock_registry.__contains__.return_value = True

    with pytest.raises(_InternalJobError) as exc_info:
        fa_job._build_dataset(DataReturnFormat.SKLEARN, ["csv"])

    assert "Cannot recover dlp" in str(exc_info.value)


def test_run_success(fa_job):
    """Test run method success."""
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": 1.5}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = fa_job.run()

        assert isinstance(reply, FAReply)
        assert reply.output == {"col1": 1.5}
        assert reply.request_id == fa_job._request_id
        assert reply.node_id == fa_job._node_id


def test_run_federated_analytics_not_allowed(fa_job_args):
    """Test run method when federated analytics is not allowed."""
    args = fa_job_args.copy()
    args["allow_fa"] = False
    job = FAJob(**args)

    reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    assert "not allowed" in reply.extra_msg


def test_run_no_dataset_args(fa_job_args):
    """Test run method when no dataset_args are provided."""
    req_args = fa_job_args["request"].__dict__.copy()
    req_args["dataset_args"] = {}
    request = FARequest(**req_args)

    args = fa_job_args.copy()
    args["request"] = request
    job = FAJob(**args)

    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": 2.0}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

        assert isinstance(reply, FAReply)
        assert reply.output == {"col1": 2.0}
        assert reply.request_id == job._request_id
        assert reply.node_id == job._node_id


def test_run_unsupported_analytics_type(
    fa_job_args, request_args, mocked_dataset_manager
):
    """Test run method with unsupported analytics type."""
    req_args = request_args.copy()
    req_args["stats"] = "unsupported_type"
    request = FARequest(**req_args)

    args = fa_job_args.copy()
    args["request"] = request
    job = FAJob(**args)

    reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    assert "unsupported values" in reply.extra_msg


def test_run_missing_compute_stats_method(fa_job):
    """Test run method when dataset does not accept compute_stats."""

    # Mock dataset without compute_stats method
    mock_dataset = MagicMock()
    del mock_dataset.compute_stats

    # Ensure hasattr(mock_dataset, 'compute_stats') returns False
    # MagicMock usually creates attributes on access, so we need to be careful.
    # Specifying spec=[] creates a mock with no attributes/methods unless added.
    mock_dataset = MagicMock(spec=[])

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = fa_job.run()

        assert isinstance(reply, ErrorMessage)
        assert reply.errnum == ErrorNumbers.FB325.value
        assert "does not support analytics method 'compute_stats'" in reply.extra_msg


def test_run_failure(fa_job):
    """Test run method when _build_dataset fails."""
    with patch.object(
        FAJob, "_build_dataset", side_effect=_InternalJobError("Dataset error")
    ):
        reply = fa_job.run()

        assert isinstance(reply, ErrorMessage)
        assert reply.errnum == ErrorNumbers.FB325.value
        assert "Dataset error" in reply.extra_msg


def test_run_compute_stats_args(fa_job):
    """Test validity of arguments passed to compute_stats."""
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"res": 1}

    # Manually set fa_args to simulate extra arguments
    fa_job._fa_args = {"some_arg_key": "some_arg_val"}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        fa_job.run()

        mock_dataset.compute_stats.assert_called_once()
        call_kwargs = mock_dataset.compute_stats.call_args[1]

        # Check if requested_stats is passed correctly (converted to list)
        assert "stats" in call_kwargs
        assert call_kwargs["stats"] == [Stats.MEAN.value]

        # Check if original fa_args are present
        assert "fa_args" in call_kwargs
        assert call_kwargs["fa_args"]["some_arg_key"] == "some_arg_val"

        # Check dataset_schema
        assert "dataset_schema" in call_kwargs
        assert call_kwargs["dataset_schema"] == ["col1", "col2"]


def test_run_compute_stats_error(fa_job):
    """Test run method when compute_stats raises an exception."""
    mock_dataset = MagicMock()
    error_msg = "Computation failed"
    mock_dataset.compute_stats.side_effect = Exception(error_msg)

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = fa_job.run()

        assert isinstance(reply, ErrorMessage)
        assert reply.errnum == ErrorNumbers.FB325.value
        assert error_msg in reply.extra_msg


def test_run_success_multiple_stats(fa_job_args, request_args):
    """Test run method with multiple requested statistics."""
    req_args = request_args.copy()
    stats_list = [Stats.MEAN.value, Stats.VARIANCE.value]
    req_args["stats"] = stats_list
    request = FARequest(**req_args)

    args = fa_job_args.copy()
    args["request"] = request
    job = FAJob(**args)

    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": {"mean": 1, "variance": 0.1}}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

        assert isinstance(reply, FAReply)
        # Check that list of stats is returned
        assert reply.stats == stats_list
        # Check passed arguments
        call_kwargs = mock_dataset.compute_stats.call_args[1]
        assert call_kwargs["stats"] == stats_list
