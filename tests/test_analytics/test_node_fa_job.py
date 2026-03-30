from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers, Stats
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.node.jobs._fa_job import FAJob, _InternalJobError

# ----------------------------- Fixtures -----------------------------


@pytest.fixture
def request_args():
    return {
        "researcher_id": "res_1",
        "request_id": "req_1",
        "experiment_id": "exp_1",
        "dataset_id": "dataset_123",
        "fa_id": "fa_1",
        "stats": [Stats.MEAN.value],
        "stats_args": {},
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
def mock_dm():
    """Minimal mocked DatasetManager with a valid dataset entry and no validate_samples side-effect."""
    dm = MagicMock()
    dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "path": "/path/to/data",
        "dataset_parameters": {},
    }
    return dm


@pytest.fixture
def mock_dataset_cls():
    """Returns a (instance, class) pair where the class always produces the same instance."""
    instance = MagicMock()

    class _MockDataset:
        def __new__(cls, *args, **kwargs):
            return instance

    return instance, _MockDataset


# ----------------------------- __init__ -----------------------------


def test_fa_job_init(fa_job, fa_job_args, request_args):
    """All constructor fields — including FAJob-specific ones — are stored correctly."""
    assert fa_job._dir == fa_job_args["root_dir"]
    assert fa_job._dataset_manager == fa_job_args["dataset_manager"]
    assert fa_job._node_id == fa_job_args["node_id"]
    assert fa_job._node_name == fa_job_args["node_name"]
    assert fa_job._dataset_id == request_args["dataset_id"]
    assert fa_job._experiment_id == request_args["experiment_id"]
    assert fa_job._fa_id == request_args["fa_id"]
    assert fa_job._researcher_id == request_args["researcher_id"]
    assert fa_job._request_id == request_args["request_id"]
    assert fa_job._stats == request_args["stats"]
    assert fa_job._stats_args == request_args["stats_args"]
    assert fa_job._dataset_schema == request_args["dataset_schema"]
    assert fa_job._allow_fa == fa_job_args["allow_fa"]


# ----------------------------- _build_error_msg ---------------------


def test_build_error_msg_explicit_errnum(fa_job):
    error = fa_job._build_error_msg("Something went wrong", ErrorNumbers.FB313.value)
    assert isinstance(error, ErrorMessage)
    assert error.extra_msg == "Something went wrong"
    assert error.errnum == ErrorNumbers.FB313.value
    assert error.node_id == fa_job._node_id
    assert error.request_id == fa_job._request_id


def test_build_error_msg_default_errnum(fa_job):
    """Default errnum should be FB313."""
    error = fa_job._build_error_msg("oops")
    assert error.errnum == ErrorNumbers.FB313.value


# ----------------------------- _build_args_for_dataset --------------


@pytest.mark.parametrize(
    "dtype_enum,entry_extra,expected",
    [
        (
            DatasetTypes.TABULAR,
            {"dtypes": {"col1": "int", "col2": "float"}},
            {"input_columns": ["col1", "col2"]},
        ),
        (
            DatasetTypes.MEDICAL_FOLDER,
            {"shape": {"t1": None, "seg": None}},
            {"data_modalities": ["t1", "seg"]},
        ),
        (DatasetTypes.IMAGES, {}, {}),
        (DatasetTypes.DEFAULT, {}, {}),
        (DatasetTypes.MEDNIST, {}, {}),
    ],
)
def test_build_args_for_dataset(fa_job, dtype_enum, entry_extra, expected):
    entry = {"data_type": dtype_enum.value, **entry_extra}
    with patch(
        "fedbiomed.node.jobs._fa_job.DatasetTypes.get_type_by_value",
        return_value=dtype_enum,
    ):
        assert fa_job._build_args_for_dataset(entry) == expected


def test_build_args_for_dataset_custom_raises(fa_job):
    entry = {"data_type": "custom"}
    with patch(
        "fedbiomed.node.jobs._fa_job.DatasetTypes.get_type_by_value",
        return_value=DatasetTypes.CUSTOM,
    ):
        with pytest.raises(_InternalJobError, match="not implemented"):
            fa_job._build_args_for_dataset(entry)


def test_build_args_for_dataset_unknown_type_raises(fa_job):
    entry = {"data_type": "unknown"}
    with patch(
        "fedbiomed.node.jobs._fa_job.DatasetTypes.get_type_by_value",
        return_value=None,
    ):
        with pytest.raises(_InternalJobError, match="unsupported dataset type"):
            fa_job._build_args_for_dataset(entry)


# ----------------------------- _build_dataset -----------------------


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
def test_build_dataset_success(mock_registry, fa_job, mock_dm, mock_dataset_cls):
    """Happy path: correct dataset returned and complete_initialization called."""
    instance, cls = mock_dataset_cls
    fa_job._dataset_manager = mock_dm
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, None, cls)

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert dataset is instance
    instance.complete_initialization.assert_called_once()
    call_args = instance.complete_initialization.call_args
    assert call_args[0][0]["root"] == "/path/to/data"
    assert call_args[0][1] == DataReturnFormat.SKLEARN


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
def test_build_dataset_forwards_dataset_parameters(
    mock_registry, fa_job, mock_dataset_cls
):
    """dataset_parameters from the DB entry are merged into controller kwargs."""
    instance, cls = mock_dataset_cls
    dm = MagicMock()
    dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "path": "/data",
        "dataset_parameters": {"tabular_file": "labels.csv"},
    }
    fa_job._dataset_manager = dm
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, None, cls)

    fa_job._build_dataset(DataReturnFormat.SKLEARN)

    call_kwargs = instance.complete_initialization.call_args[0][0]
    assert call_kwargs.get("tabular_file") == "labels.csv"


def test_build_dataset_not_found(fa_job, mock_dm):
    """_build_dataset raises when dataset_id is absent from the DB."""
    mock_dm.dataset_table.get_by_id.return_value = None
    fa_job._dataset_manager = mock_dm

    with pytest.raises(_InternalJobError, match="Cannot find requested dataset"):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS", {})
def test_build_dataset_unsupported_type(fa_job, mock_dm):
    """_build_dataset raises when data_type is not in REGISTRY_CONTROLLERS."""
    mock_dm.dataset_table.get_by_id.return_value = {"data_type": "unknown_type"}
    fa_job._dataset_manager = mock_dm

    with pytest.raises(_InternalJobError, match="not supported"):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
@patch("fedbiomed.node.jobs._fa_job.DataLoadingPlan")
def test_build_dataset_with_dlp(
    mock_dlp_cls, mock_registry, fa_job, mock_dm, mock_dataset_cls
):
    """DLP is deserialised and passed as 'dlp' in controller kwargs."""
    instance, cls = mock_dataset_cls
    mock_dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "path": "/path",
        "dlp_id": "dlp_1",
    }
    mock_dm.get_dlp_by_id.return_value = ["dlp_content"]
    fa_job._dataset_manager = mock_dm

    mock_dlp = mock_dlp_cls.return_value
    mock_dlp.deserialize.return_value = mock_dlp
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, None, cls)

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert dataset is instance
    assert instance.complete_initialization.call_args[0][0]["dlp"] == mock_dlp


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
@patch("fedbiomed.node.jobs._fa_job.DataLoadingPlan")
def test_build_dataset_dlp_deserialisation_error(
    mock_dlp_cls, mock_registry, fa_job, mock_dm
):
    """Failed DLP deserialisation is wrapped as _InternalJobError."""
    mock_dm.dataset_table.get_by_id.return_value = {
        "data_type": "csv",
        "dlp_id": "dlp_1",
    }
    mock_dm.get_dlp_by_id.return_value = ["dlp_content"]
    fa_job._dataset_manager = mock_dm
    mock_dlp_cls.return_value.deserialize.side_effect = FedbiomedError("DLP Error")
    mock_registry.__contains__.return_value = True

    with pytest.raises(_InternalJobError, match="Cannot recover dlp"):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
def test_build_dataset_initialization_error(mock_registry, fa_job, mock_dm):
    """FedbiomedError during dataset construction is wrapped as _InternalJobError."""
    fa_job._dataset_manager = mock_dm
    broken_cls = MagicMock(side_effect=FedbiomedError("Init Error"))
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, None, broken_cls)

    with pytest.raises(
        _InternalJobError, match="Cannot initialize dataset.*Init Error"
    ):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
def test_build_dataset_below_minimum_samples_raises(
    mock_registry, fa_job, mock_dm, mock_dataset_cls
):
    """validate_samples failure is wrapped as _InternalJobError."""
    _, cls = mock_dataset_cls
    mock_dm.validate_samples.side_effect = FedbiomedError("below minimum")
    fa_job._dataset_manager = mock_dm
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, None, cls)

    with pytest.raises(
        _InternalJobError, match="does not meet minimum sample requirement"
    ):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


@patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
def test_build_dataset_validate_samples_called(
    mock_registry, fa_job, mock_dm, mock_dataset_cls
):
    """validate_samples is always called after the dataset is built."""
    _, cls = mock_dataset_cls
    fa_job._dataset_manager = mock_dm
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, None, cls)

    fa_job._build_dataset(DataReturnFormat.SKLEARN)

    mock_dm.validate_samples.assert_called_once()


# ----------------------------- run() --------------------------------


def test_run_success(fa_job):
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": 1.5}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = fa_job.run()

    assert isinstance(reply, FAReply)
    assert reply.output == {"col1": 1.5}
    assert reply.request_id == fa_job._request_id
    assert reply.node_id == fa_job._node_id
    assert reply.experiment_id == fa_job._experiment_id
    assert reply.fa_id == fa_job._fa_id


def test_run_fa_not_allowed(fa_job_args):
    fa_job_args["allow_fa"] = False
    job = FAJob(**fa_job_args)
    reply = job.run()
    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    assert "not allowed" in reply.extra_msg


def test_run_no_stats_provided(fa_job_args, request_args):
    """Both stats=None and stats_args={} triggers the validation error."""
    req = FARequest(**{**request_args, "stats": None, "stats_args": {}})
    job = FAJob(**{**fa_job_args, "request": req})
    reply = job.run()
    assert isinstance(reply, ErrorMessage)
    assert "At least one of 'stats' or 'stats_args' must be provided" in reply.extra_msg


def test_run_only_stats_args_succeeds(fa_job_args, request_args):
    """stats=None is valid when stats_args is non-empty."""
    req = FARequest(
        **{**request_args, "stats": None, "stats_args": {Stats.MEAN.value: {}}}
    )
    job = FAJob(**{**fa_job_args, "request": req})
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": 1.0}
    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()
    assert isinstance(reply, FAReply)


def test_run_invalid_stats_values(fa_job_args, request_args):
    req = FARequest(**{**request_args, "stats": ["unsupported_stat"]})
    job = FAJob(**{**fa_job_args, "request": req})
    reply = job.run()
    assert isinstance(reply, ErrorMessage)
    assert "unsupported values" in reply.extra_msg


def test_run_invalid_stats_args_keys(fa_job_args, request_args):
    req = FARequest(**{**request_args, "stats": None, "stats_args": {"bad_key": {}}})
    job = FAJob(**{**fa_job_args, "request": req})
    reply = job.run()
    assert isinstance(reply, ErrorMessage)
    assert "contains unsupported keys" in reply.extra_msg


def test_run_build_dataset_fails(fa_job):
    with patch.object(
        FAJob, "_build_dataset", side_effect=_InternalJobError("Dataset error")
    ):
        reply = fa_job.run()
    assert isinstance(reply, ErrorMessage)
    assert "Dataset error" in reply.extra_msg


def test_run_missing_compute_stats(fa_job):
    mock_dataset = MagicMock(spec=[])  # no attributes at all
    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = fa_job.run()
    assert isinstance(reply, ErrorMessage)
    assert "does not support analytics method 'compute_stats'" in reply.extra_msg


def test_run_compute_stats_raises(fa_job):
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.side_effect = Exception("Computation failed")
    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = fa_job.run()
    assert isinstance(reply, ErrorMessage)
    assert "Computation failed" in reply.extra_msg


def test_run_passes_correct_kwargs_to_compute_stats(fa_job):
    """stats, stats_args, and dataset_schema are forwarded unchanged."""
    fa_job._stats_args = {Stats.MEAN.value: {"some_arg": "val"}}
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"res": 1}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        fa_job.run()

    kwargs = mock_dataset.compute_stats.call_args[1]
    assert kwargs["stats"] == [Stats.MEAN.value]
    assert kwargs["stats_args"][Stats.MEAN.value]["some_arg"] == "val"
    assert kwargs["dataset_schema"] == ["col1", "col2"]


def test_run_multiple_stats(fa_job_args, request_args):
    stats_list = [Stats.MEAN.value, Stats.VARIANCE.value]
    req = FARequest(**{**request_args, "stats": stats_list})
    job = FAJob(**{**fa_job_args, "request": req})
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": {"mean": 1, "variance": 0.1}}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, FAReply)
    assert reply.stats == stats_list
    assert mock_dataset.compute_stats.call_args[1]["stats"] == stats_list
