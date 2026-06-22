from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import (
    DatasetTypes,
    ErrorNumbers,
    SAParameters,
    Stats,
)
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedSecureAggregationError
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
            {"dtypes": {"col1": "Int64", "col2": "Float64"}},
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


@pytest.mark.parametrize(
    "dtypes,expected_columns",
    [
        # All numerical
        (
            {"a": "Int8", "b": "Int16", "c": "Int32", "d": "Int64", "e": "Int128"},
            ["a", "b", "c", "d", "e"],
        ),
        (
            {"a": "UInt8", "b": "UInt16", "c": "UInt32", "d": "UInt64"},
            ["a", "b", "c", "d"],
        ),
        ({"a": "Float32", "b": "Float64"}, ["a", "b"]),
        # Mixed: only numerical columns returned
        ({"num": "Int64", "text": "String", "flag": "Boolean", "dt": "Date"}, ["num"]),
        ({"x": "Float32", "label": "String"}, ["x"]),
        # All non-numerical: empty list
        ({"a": "String", "b": "Boolean", "c": "Date"}, []),
        # Empty dtypes: empty list
        ({}, []),
    ],
)
def test_build_args_for_dataset_tabular_numerical_filter(
    fa_job, dtypes, expected_columns
):
    """Only columns with numerical polars dtypes are included in input_columns."""
    entry = {"data_type": DatasetTypes.TABULAR.value, "dtypes": dtypes}
    with patch(
        "fedbiomed.node.jobs._fa_job.DatasetTypes.get_type_by_value",
        return_value=DatasetTypes.TABULAR,
    ):
        result = fa_job._build_args_for_dataset(entry)
    assert result["input_columns"] == expected_columns


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
    """Happy path: correct dataset returned and load called."""
    instance, cls = mock_dataset_cls
    fa_job._dataset_manager = mock_dm
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (None, cls)

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert dataset is instance
    instance.load.assert_called_once()
    call_args = instance.load.call_args
    assert call_args.kwargs["root"] == "/path/to/data"
    assert call_args.kwargs["to_format"] == DataReturnFormat.SKLEARN


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
    mock_registry.__getitem__.return_value = (None, cls)

    fa_job._build_dataset(DataReturnFormat.SKLEARN)

    call_kwargs = instance.load.call_args.kwargs
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
    mock_registry.__getitem__.return_value = (None, cls)

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert dataset is instance
    assert instance.load.call_args.kwargs["dlp"] == mock_dlp


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
    mock_registry.__getitem__.return_value = (None, broken_cls)

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
    mock_registry.__getitem__.return_value = (None, cls)

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
    mock_registry.__getitem__.return_value = (None, cls)

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


def test_run_only_stats_args_succeeds(fa_job_args, request_args):
    """stats=None is valid when stats_args carries the nested schema-key structure."""
    req = FARequest(
        **{
            **request_args,
            "stats": None,
            "stats_args": {"price": {Stats.MEAN.value: {}}},
            "dataset_schema": None,
        }
    )
    job = FAJob(**{**fa_job_args, "request": req})
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"price": 1.0}
    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()
    assert isinstance(reply, FAReply)


def test_run_invalid_stats_values(fa_job_args, request_args):
    req = FARequest(**{**request_args, "stats": ["unsupported_stat"]})
    job = FAJob(**{**fa_job_args, "request": req})
    reply = job.run()
    assert isinstance(reply, ErrorMessage)
    assert "unsupported values" in reply.extra_msg


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
    fa_job._stats_args = {"price": {Stats.MEAN.value: {"some_arg": "val"}}}
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"res": 1}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        fa_job.run()

    kwargs = mock_dataset.compute_stats.call_args[1]
    assert kwargs["stats"] == [Stats.MEAN.value]
    assert kwargs["stats_args"]["price"][Stats.MEAN.value]["some_arg"] == "val"
    assert kwargs["dataset_schema"] == ["col1", "col2"]


def test_run_multiple_stats(fa_job_args, request_args):
    stats_list = [Stats.MEAN.value, Stats.SUM.value]
    req = FARequest(**{**request_args, "stats": stats_list})
    job = FAJob(**{**fa_job_args, "request": req})
    mock_dataset = MagicMock()
    # A [mean, sum] request yields the count+sum primitives (mean -> count+sum).
    mock_dataset.compute_stats.return_value = {"col1": {"count": 6, "sum": 6}}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, FAReply)
    assert reply.stats == stats_list
    assert mock_dataset.compute_stats.call_args[1]["stats"] == stats_list


# ----------------------------- run() — secagg paths -----------------


@pytest.fixture
def secagg_args():
    """Minimal secagg_arguments dict for a JLS FA round."""
    from fedbiomed.common.constants import SecureAggregationSchemes

    return {
        "secagg_scheme": SecureAggregationSchemes.JOYE_LIBERT,
        "secagg_servkey_id": "serv-id",
        "secagg_random": 0.5,
        "parties": ["researcher-1", "node-1", "node-2"],
        "fa_round": 2,
    }


def _make_fa_job(
    fa_job_args,
    fa_request,
    secagg_active=False,
    force_secagg=False,
    secagg_arguments=None,
):
    """Helper: build FAJob with secagg parameters."""
    return FAJob(
        **{
            **fa_job_args,
            "request": fa_request,
            "db": "/tmp/test.json",
            "secagg_active": secagg_active,
            "force_secagg": force_secagg,
            "secagg_arguments": secagg_arguments,
        }
    )


def test_run_secagg_force_without_args_returns_error(fa_job_args, fa_request):
    """force_secagg=True + no secagg_arguments → ErrorMessage (node policy violated)."""
    job = _make_fa_job(
        fa_job_args,
        fa_request,
        secagg_active=True,
        force_secagg=True,
        secagg_arguments=None,
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": 1.0}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value


@patch("fedbiomed.node.jobs._fa_job.SecaggRound")
def test_run_secagg_request_with_clipping_range_ignored(
    mock_secagg_cls, fa_job_args, fa_request, secagg_args
):
    """A request carrying secagg_clipping_range is not rejected; the node ignores it and uses FA_CLIPPING_RANGE."""
    mock_secagg = MagicMock()
    mock_secagg.use_secagg = True
    mock_secagg.scheme.encrypt.return_value = [1]
    mock_secagg_cls.return_value = mock_secagg

    secagg_args["secagg_clipping_range"] = 3
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"x": 1.0}  # within FA_CLIPPING_RANGE

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert not isinstance(reply, ErrorMessage)
    # The request value (3) is ignored in favour of the fixed node-side constant.
    call_args = mock_secagg.scheme.encrypt.call_args
    assert call_args.kwargs["clipping_range"] == SAParameters.FA_CLIPPING_RANGE


def test_run_secagg_not_active_returns_error(fa_job_args, fa_request, secagg_args):
    """secagg_arguments provided + secagg_active=False → ErrorMessage."""
    job = _make_fa_job(
        fa_job_args,
        fa_request,
        secagg_active=False,
        force_secagg=False,
        secagg_arguments=secagg_args,
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"col1": 1.0}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value


@patch("fedbiomed.node.jobs._fa_job.SecaggRound")
def test_run_encrypted_path_returns_fa_reply(
    mock_secagg_cls, fa_job_args, fa_request, secagg_args
):
    """When use_secagg is True, run() returns encrypted FAReply."""
    mock_secagg = MagicMock()
    mock_secagg.use_secagg = True
    mock_secagg.scheme.encrypt.return_value = [100, 200, 300]
    mock_secagg_cls.return_value = mock_secagg

    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"a": 1.0, "b": 2.0, "c": 3.0}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, FAReply)
    assert reply.encrypted is True
    assert reply.params_encrypted == [100, 200, 300]
    assert reply.output is None
    assert reply.output_schema is not None
    assert mock_secagg.scheme.encrypt.call_count == 1


@patch("fedbiomed.node.jobs._fa_job.SecaggRound")
def test_run_encrypted_path_rejects_out_of_clipping_range(
    mock_secagg_cls, fa_job_args, fa_request, secagg_args
):
    """A statistic beyond the clipping range yields an ErrorMessage, not encryption."""
    mock_secagg = MagicMock()
    mock_secagg.use_secagg = True
    mock_secagg_cls.return_value = mock_secagg

    # a statistic above the fixed node-side clipping range is rejected
    over = SAParameters.FA_CLIPPING_RANGE + 1
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {
        "age": {"sum": 1.0, "count": 2},
        "income": {"sum": over, "count": 2},  # exceeds FA_CLIPPING_RANGE; 'age' is fine
    }

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    mock_secagg.scheme.encrypt.assert_not_called()
    # The message names the offending column/stat to help the user...
    assert "income.sum" in reply.extra_msg
    # ...but never echoes the offending value (would leak an unencrypted statistic).
    assert str(over) not in reply.extra_msg


@patch("fedbiomed.node.jobs._fa_job.SecaggRound")
def test_run_encrypted_path_rejects_out_of_clipping_range_unnamed(
    mock_secagg_cls, fa_job_args, fa_request, secagg_args
):
    """A bare-scalar leaf has no key-path: the error still fires, without a name."""
    mock_secagg = MagicMock()
    mock_secagg.use_secagg = True
    mock_secagg_cls.return_value = mock_secagg

    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    mock_dataset = MagicMock()
    # scalar → empty key-path; exceeds the fixed FA_CLIPPING_RANGE constant
    mock_dataset.compute_stats.return_value = float(SAParameters.FA_CLIPPING_RANGE + 1)

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    mock_secagg.scheme.encrypt.assert_not_called()
    # No name available → no "Offending statistic(s)" clause, just the generic error.
    assert "Offending statistic(s)" not in reply.extra_msg
    assert "exceed the secure aggregation clipping range" in reply.extra_msg


@patch("fedbiomed.node.jobs._fa_job.SecaggRound")
def test_run_encrypted_path_uses_fa_round_from_args(
    mock_secagg_cls, fa_job_args, fa_request, secagg_args
):
    """encrypt() is called with the fa_round from secagg_arguments."""
    mock_secagg = MagicMock()
    mock_secagg.use_secagg = True
    mock_secagg.scheme.encrypt.return_value = [1]
    mock_secagg_cls.return_value = mock_secagg

    secagg_args["fa_round"] = 7
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"x": 1.0}  # within FA_CLIPPING_RANGE

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        job.run()

    call_args = mock_secagg.scheme.encrypt.call_args
    assert call_args.kwargs["current_round"] == 7
    # FA clipping range is the fixed node-side constant, not the request value
    assert call_args.kwargs["clipping_range"] == SAParameters.FA_CLIPPING_RANGE


@patch("fedbiomed.node.jobs._fa_job.SecaggRound")
def test_run_secagg_round_error_returns_error_message(
    mock_secagg_cls, fa_job_args, fa_request, secagg_args
):
    """FedbiomedSecureAggregationError from SecaggRound → ErrorMessage."""
    mock_secagg_cls.side_effect = FedbiomedSecureAggregationError("context missing")

    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    mock_dataset = MagicMock()
    mock_dataset.compute_stats.return_value = {"a": 1.0}

    with patch.object(FAJob, "_build_dataset", return_value=mock_dataset):
        reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    assert "context missing" in reply.extra_msg


# ----------------------------- _check_clipping_overflow -------------


@pytest.mark.parametrize(
    "flat,schema,clip",
    [
        ([], [], 3),  # no values
        ([3, -3, 0], [["a"], ["b"], ["c"]], 3),  # exactly on the bounds is allowed
        ([1.5, -2.0], [["a", "sum"], ["b"]], 3),  # strictly inside
    ],
)
def test_check_clipping_overflow_within_range_returns_none(fa_job, flat, schema, clip):
    """Values in [-clip, clip] (bounds inclusive) produce no error."""
    assert fa_job._check_clipping_overflow(flat, schema, clip) is None


def test_check_clipping_overflow_over_upper_bound(fa_job):
    """A value above clip is reported by its key-path, count is 1."""
    err = fa_job._check_clipping_overflow([1.0, 99.0], [["age"], ["income", "sum"]], 3)
    assert isinstance(err, ErrorMessage)
    assert err.errnum == ErrorNumbers.FB325.value
    assert "1 computed analytics value(s)" in err.extra_msg
    assert "income.sum" in err.extra_msg
    assert "99" not in err.extra_msg  # never echo the unencrypted value


def test_check_clipping_overflow_below_lower_bound(fa_job):
    """A value below -clip is also caught."""
    err = fa_job._check_clipping_overflow([-99.0], [["loss"]], 3)
    assert isinstance(err, ErrorMessage)
    assert "loss" in err.extra_msg


def test_check_clipping_overflow_unnamed_offender(fa_job):
    """A bare scalar (empty key-path) fires the error without a name clause."""
    err = fa_job._check_clipping_overflow([99.0], [[]], 3)
    assert isinstance(err, ErrorMessage)
    assert "Offending statistic(s)" not in err.extra_msg
    assert "exceed the secure aggregation clipping range" in err.extra_msg


def test_check_clipping_overflow_multiple_offenders_named_and_unnamed(fa_job):
    """Count tallies every offender; only those with a key-path are named."""
    err = fa_job._check_clipping_overflow(
        [99.0, 1.0, -50.0, 99.0],
        [["a", "sum"], ["b"], [], ["c"]],
        3,
    )
    assert "3 computed analytics value(s)" in err.extra_msg
    assert "a.sum" in err.extra_msg and "c" in err.extra_msg
    assert "b" not in err.extra_msg.split("Offending statistic(s):")[1]
