from unittest.mock import MagicMock

import pytest
from fedbiomed.common.constants import (
    DatasetTypes,
    ErrorNumbers,
    SAParameters,
    SecureAggregationSchemes,
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
    dm.get_dataset.return_value = {
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


@pytest.fixture
def dataset_type(mocker):
    """Makes the dataset entry read as the given type."""
    return lambda dtype: mocker.patch(
        "fedbiomed.node.jobs._fa_job.DatasetTypes.get_type_by_value",
        return_value=dtype,
    )


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
def test_build_args_for_dataset(
    fa_job, dataset_type, dtype_enum, entry_extra, expected
):
    dataset_type(dtype_enum)

    entry = {"data_type": dtype_enum.value, **entry_extra}
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
    fa_job, dataset_type, dtypes, expected_columns
):
    """Only columns with numerical polars dtypes are included in input_columns."""
    dataset_type(DatasetTypes.TABULAR)

    entry = {"data_type": DatasetTypes.TABULAR.value, "dtypes": dtypes}
    assert fa_job._build_args_for_dataset(entry)["input_columns"] == expected_columns


@pytest.mark.parametrize(
    "dtype_enum, data_type, refusal",
    [
        (DatasetTypes.CUSTOM, "custom", "not implemented"),
        (None, "unknown", "unsupported dataset type"),
    ],
    ids=["custom", "unknown"],
)
def test_build_args_for_dataset_refuses_type(
    fa_job, dataset_type, dtype_enum, data_type, refusal
):
    dataset_type(dtype_enum)

    with pytest.raises(_InternalJobError, match=refusal):
        fa_job._build_args_for_dataset({"data_type": data_type})


# ----------------------------- _build_dataset -----------------------


@pytest.fixture
def registry(mocker, fa_job, mock_dm, mock_dataset_cls):
    """`fa_job` on `mock_dm`, with every data type built by `mock_dataset_cls`."""
    _, cls = mock_dataset_cls
    fa_job._dataset_manager = mock_dm
    registry = mocker.patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS")
    registry.__contains__.return_value = True
    registry.__getitem__.return_value = (None, cls)
    return registry


def test_build_dataset_success(fa_job, registry, mock_dataset_cls):
    """Happy path: correct dataset returned and load called."""
    instance, _ = mock_dataset_cls

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert dataset is instance
    instance.load.assert_called_once()
    call_args = instance.load.call_args
    assert call_args.kwargs["root"] == "/path/to/data"
    assert call_args.kwargs["to_format"] == DataReturnFormat.SKLEARN


def test_build_dataset_forwards_dataset_parameters(
    fa_job, registry, mock_dm, mock_dataset_cls
):
    """dataset_parameters from the DB entry are merged into controller kwargs."""
    instance, _ = mock_dataset_cls
    mock_dm.get_dataset.return_value = {
        "data_type": "csv",
        "path": "/data",
        "dataset_parameters": {"tabular_file": "labels.csv"},
    }

    fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert instance.load.call_args.kwargs.get("tabular_file") == "labels.csv"


def test_build_dataset_not_found(fa_job, mock_dm):
    """_build_dataset raises when dataset_id is absent from the DB."""
    mock_dm.get_dataset.return_value = None
    fa_job._dataset_manager = mock_dm

    with pytest.raises(_InternalJobError, match="Cannot find requested dataset"):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


def test_build_dataset_unsupported_type(mocker, fa_job, mock_dm):
    """_build_dataset raises when data_type is not in REGISTRY_CONTROLLERS."""
    mocker.patch("fedbiomed.node.jobs._fa_job.REGISTRY_CONTROLLERS", {})
    mock_dm.get_dataset.return_value = {"data_type": "unknown_type"}
    fa_job._dataset_manager = mock_dm

    with pytest.raises(_InternalJobError, match="not supported"):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


def test_build_dataset_with_dlp(mocker, fa_job, registry, mock_dm, mock_dataset_cls):
    """DLP is deserialised and passed as 'dlp' in controller kwargs."""
    instance, _ = mock_dataset_cls
    mock_dm.get_dataset.return_value = {
        "data_type": "csv",
        "path": "/path",
        "dlp_id": "dlp_1",
    }
    mock_dm.get_dlp_by_id.return_value = ["dlp_content"]
    mock_dlp = mocker.patch("fedbiomed.node.jobs._fa_job.DataLoadingPlan").return_value
    mock_dlp.deserialize.return_value = mock_dlp

    dataset = fa_job._build_dataset(DataReturnFormat.SKLEARN)

    assert dataset is instance
    assert instance.load.call_args.kwargs["dlp"] == mock_dlp


def test_build_dataset_dlp_deserialisation_error(mocker, fa_job, registry, mock_dm):
    """Failed DLP deserialisation is wrapped as _InternalJobError."""
    mock_dm.get_dataset.return_value = {"data_type": "csv", "dlp_id": "dlp_1"}
    mock_dm.get_dlp_by_id.return_value = ["dlp_content"]
    dlp_class = mocker.patch("fedbiomed.node.jobs._fa_job.DataLoadingPlan")
    dlp_class.return_value.deserialize.side_effect = FedbiomedError("DLP Error")

    with pytest.raises(_InternalJobError, match="Cannot recover dlp"):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


def test_build_dataset_initialization_error(fa_job, registry):
    """FedbiomedError during dataset construction is wrapped as _InternalJobError."""
    registry.__getitem__.return_value = (
        None,
        MagicMock(side_effect=FedbiomedError("Init Error")),
    )

    with pytest.raises(
        _InternalJobError, match="Cannot initialize dataset.*Init Error"
    ):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


def test_build_dataset_below_minimum_samples_raises(fa_job, registry, mock_dm):
    """validate_samples failure is wrapped as _InternalJobError."""
    mock_dm.validate_samples.side_effect = FedbiomedError("below minimum")

    with pytest.raises(
        _InternalJobError, match="does not meet minimum sample requirement"
    ):
        fa_job._build_dataset(DataReturnFormat.SKLEARN)


def test_build_dataset_validate_samples_called(fa_job, registry, mock_dm):
    """validate_samples is always called after the dataset is built."""
    fa_job._build_dataset(DataReturnFormat.SKLEARN)

    mock_dm.validate_samples.assert_called_once()


# ----------------------------- run() --------------------------------


@pytest.fixture
def dataset(mocker):
    """The dataset the job builds, mocked: tests set what `compute_stats` returns."""
    return mocker.patch.object(FAJob, "_build_dataset").return_value


def test_run_success(fa_job, dataset):
    dataset.compute_stats.return_value = {"col1": 1.5}

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


def test_run_only_stats_args_succeeds(fa_job_args, request_args, dataset):
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
    dataset.compute_stats.return_value = {"price": 1.0}

    assert isinstance(job.run(), FAReply)


def test_run_invalid_stats_values(fa_job_args, request_args):
    req = FARequest(**{**request_args, "stats": ["unsupported_stat"]})
    job = FAJob(**{**fa_job_args, "request": req})
    reply = job.run()
    assert isinstance(reply, ErrorMessage)
    assert "unsupported values" in reply.extra_msg


@pytest.mark.parametrize(
    "built, refusal",
    [
        ({"side_effect": _InternalJobError("Dataset error")}, "Dataset error"),
        (
            # no attributes at all
            {"return_value": MagicMock(spec=[])},
            "does not support analytics method 'compute_stats'",
        ),
        (
            {
                "return_value": MagicMock(
                    **{"compute_stats.side_effect": Exception("Computation failed")}
                )
            },
            "Computation failed",
        ),
    ],
    ids=["build_fails", "no_compute_stats", "compute_stats_raises"],
)
def test_run_replies_with_an_error_when_the_dataset_fails(
    mocker, fa_job, built, refusal
):
    mocker.patch.object(FAJob, "_build_dataset", **built)

    reply = fa_job.run()

    assert isinstance(reply, ErrorMessage)
    assert refusal in reply.extra_msg


def test_run_passes_correct_kwargs_to_compute_stats(fa_job, dataset):
    """stats, stats_args, and dataset_schema are forwarded unchanged."""
    fa_job._stats_args = {"price": {Stats.MEAN.value: {"some_arg": "val"}}}
    dataset.compute_stats.return_value = {"res": 1}

    fa_job.run()

    kwargs = dataset.compute_stats.call_args[1]
    assert kwargs["stats"] == [Stats.MEAN.value]
    assert kwargs["stats_args"]["price"][Stats.MEAN.value]["some_arg"] == "val"
    assert kwargs["dataset_schema"] == ["col1", "col2"]


def test_run_multiple_stats(fa_job_args, request_args, dataset):
    stats_list = [Stats.MEAN.value, Stats.SUM.value]
    req = FARequest(**{**request_args, "stats": stats_list})
    job = FAJob(**{**fa_job_args, "request": req})
    # A [mean, sum] request yields the count+sum primitives (mean -> count+sum).
    dataset.compute_stats.return_value = {"col1": {"count": 6, "sum": 6}}

    reply = job.run()

    assert isinstance(reply, FAReply)
    assert reply.stats == stats_list
    assert dataset.compute_stats.call_args[1]["stats"] == stats_list


# ----------------------------- run() — secagg paths -----------------


@pytest.fixture
def secagg_args():
    """Minimal secagg_arguments dict for a JLS FA round."""
    return {
        "secagg_scheme": SecureAggregationSchemes.JOYE_LIBERT,
        "secagg_servkey_id": "serv-id",
        "secagg_random": 0.5,
        "parties": ["researcher-1", "node-1", "node-2"],
        "fa_round": 2,
    }


@pytest.fixture
def secagg(mocker):
    """The secure aggregation round the job opens, mocked: in use, encrypting to [1]."""
    secagg_round = mocker.patch("fedbiomed.node.jobs._fa_job.SecaggRound").return_value
    secagg_round.use_secagg = True
    secagg_round.scheme.encrypt.return_value = [1]
    return secagg_round


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


@pytest.mark.parametrize(
    "secagg_active, force_secagg, with_arguments",
    [(True, True, False), (False, False, True)],
    ids=["forced_without_arguments", "arguments_while_not_active"],
)
def test_run_refuses_secagg_against_the_node_policy(
    fa_job_args,
    fa_request,
    secagg_args,
    dataset,
    secagg_active,
    force_secagg,
    with_arguments,
):
    job = _make_fa_job(
        fa_job_args,
        fa_request,
        secagg_active=secagg_active,
        force_secagg=force_secagg,
        secagg_arguments=secagg_args if with_arguments else None,
    )
    dataset.compute_stats.return_value = {"col1": 1.0}

    reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value


@pytest.mark.parametrize(
    "fa_round, request_clipping_range",
    [(2, 3), (7, None)],
    ids=["request_clipping_range_ignored", "round_from_arguments"],
)
def test_run_encrypts_with_the_round_and_the_node_clipping_range(
    fa_job_args,
    fa_request,
    secagg_args,
    dataset,
    secagg,
    fa_round,
    request_clipping_range,
):
    """A request's secagg_clipping_range is not rejected but ignored for FA_CLIPPING_RANGE."""
    secagg_args["fa_round"] = fa_round
    if request_clipping_range is not None:
        secagg_args["secagg_clipping_range"] = request_clipping_range
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    dataset.compute_stats.return_value = {"x": 1.0}  # within FA_CLIPPING_RANGE

    reply = job.run()

    assert not isinstance(reply, ErrorMessage)
    call_args = secagg.scheme.encrypt.call_args
    assert call_args.kwargs["current_round"] == fa_round
    assert call_args.kwargs["clipping_range"] == SAParameters.FA_CLIPPING_RANGE


def test_run_encrypted_path_returns_fa_reply(
    fa_job_args, fa_request, secagg_args, dataset, secagg
):
    """When use_secagg is True, run() returns encrypted FAReply."""
    secagg.scheme.encrypt.return_value = [100, 200, 300]
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    dataset.compute_stats.return_value = {"a": 1.0, "b": 2.0, "c": 3.0}

    reply = job.run()

    assert isinstance(reply, FAReply)
    assert reply.encrypted is True
    assert reply.params_encrypted == [100, 200, 300]
    assert reply.output is None
    assert reply.output_schema is not None
    assert secagg.scheme.encrypt.call_count == 1


def test_run_encrypted_path_rejects_out_of_clipping_range(
    fa_job_args, fa_request, secagg_args, dataset, secagg
):
    """A statistic beyond the clipping range yields an ErrorMessage, not encryption."""
    # a statistic above the fixed node-side clipping range is rejected
    over = SAParameters.FA_CLIPPING_RANGE + 1
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    dataset.compute_stats.return_value = {
        "age": {"sum": 1.0, "count": 2},
        "income": {"sum": over, "count": 2},  # exceeds FA_CLIPPING_RANGE; 'age' is fine
    }

    reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    secagg.scheme.encrypt.assert_not_called()
    # The message names the offending column/stat to help the user...
    assert "income.sum" in reply.extra_msg
    # ...but never echoes the offending value (would leak an unencrypted statistic).
    assert str(over) not in reply.extra_msg


def test_run_encrypted_path_rejects_out_of_clipping_range_unnamed(
    fa_job_args, fa_request, secagg_args, dataset, secagg
):
    """A bare-scalar leaf has no key-path: the error still fires, without a name."""
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    # scalar → empty key-path; exceeds the fixed FA_CLIPPING_RANGE constant
    dataset.compute_stats.return_value = float(SAParameters.FA_CLIPPING_RANGE + 1)

    reply = job.run()

    assert isinstance(reply, ErrorMessage)
    assert reply.errnum == ErrorNumbers.FB325.value
    secagg.scheme.encrypt.assert_not_called()
    # No name available → no "Offending statistic(s)" clause, just the generic error.
    assert "Offending statistic(s)" not in reply.extra_msg
    assert "exceed the secure aggregation clipping range" in reply.extra_msg


def test_run_secagg_round_error_returns_error_message(
    mocker, fa_job_args, fa_request, secagg_args, dataset
):
    """FedbiomedSecureAggregationError from SecaggRound → ErrorMessage."""
    mocker.patch(
        "fedbiomed.node.jobs._fa_job.SecaggRound",
        side_effect=FedbiomedSecureAggregationError("context missing"),
    )
    job = _make_fa_job(
        fa_job_args, fa_request, secagg_active=True, secagg_arguments=secagg_args
    )
    dataset.compute_stats.return_value = {"a": 1.0}

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
