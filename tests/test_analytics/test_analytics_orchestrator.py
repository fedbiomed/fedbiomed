from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.analytics import AnalyticsOrchestrator
from fedbiomed.common.analytics.accumulators import (
    DictAccumulator,
    ImageAccumulator,
    RowAccumulator,
    SequenceAccumulator,
)
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetElementType,
    ImageSpec,
    RowSpec,
)
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def orchestrator():
    return AnalyticsOrchestrator()


@pytest.fixture
def mock_dataset():
    d = MagicMock()
    d.__len__.return_value = 10
    d.__getitem__.return_value = {"a": [1, 2]}
    # Make it iterable like a Dataset
    d.__iter__.return_value = iter([{"a": [1, 2]}] * 10)
    d.analytics_schema.return_value = {"a": RowSpec(columns=["col1", "col2"])}
    d.to_format = DataReturnFormat.SKLEARN
    return d


# ── compute_stats ────────────────────────────────────────────────────────────


def test_compute_stats_missing_capability(orchestrator):
    dataset = MagicMock()
    del dataset.analytics_schema
    dataset.to_format = DataReturnFormat.SKLEARN

    with pytest.raises(FedbiomedError, match="Dataset does not implement"):
        orchestrator.compute_stats(dataset, stats=["mean"])


def test_compute_stats_no_stats_no_stats_args_raises(orchestrator, mock_dataset):
    """Both stats and stats_args being empty/None must raise immediately."""
    with pytest.raises(FedbiomedError, match="At least one of 'stats' or 'stats_args'"):
        orchestrator.compute_stats(mock_dataset)

    with pytest.raises(FedbiomedError, match="At least one of 'stats' or 'stats_args'"):
        orchestrator.compute_stats(mock_dataset, stats=[], stats_args={})


def test_compute_stats_invalid_format(orchestrator, mock_dataset):
    mock_dataset.to_format = DataReturnFormat.TORCH
    with pytest.raises(FedbiomedError, match="Dataset format: '.*' is not supported"):
        orchestrator.compute_stats(mock_dataset, stats=["mean"])


@patch.object(AnalyticsOrchestrator, "_create_accumulator")
def test_compute_stats_flow(mock_create_acc, orchestrator, mock_dataset):
    mock_acc = MagicMock()
    mock_create_acc.return_value = mock_acc
    mock_acc.finalize.return_value = "results"

    res = orchestrator.compute_stats(mock_dataset, stats=["mean"])

    mock_dataset.analytics_schema.assert_called_once()
    mock_create_acc.assert_called_once()
    assert mock_acc.update.call_count == 10
    mock_acc.finalize.assert_called_once()
    assert res == "results"


# ── _build_and_validate_config dispatch ─────────────────────────────────────


def test_build_and_validate_config_dispatch_dict(orchestrator):
    schema = {"a": ImageSpec()}
    with patch.object(orchestrator, "_handle_dict") as mock_hd:
        orchestrator._build_and_validate_config(schema, None, None, None, 5)
        mock_hd.assert_called_once_with(schema, None, None, None, 5)


@pytest.mark.parametrize("schema", [[ImageSpec()], (ImageSpec(),)])
def test_build_and_validate_config_dispatch_sequence(schema, orchestrator):
    with patch.object(orchestrator, "_handle_sequence") as mock_hs:
        orchestrator._build_and_validate_config(schema, None, None, None, 5)
        mock_hs.assert_called_once_with(schema, None, None, None, 5)


def test_build_and_validate_config_dispatch_row(orchestrator):
    schema = RowSpec(columns=["c1"])
    with patch.object(orchestrator, "_handle_row") as mock_hr:
        orchestrator._build_and_validate_config(schema, None, None, None, 5)
        mock_hr.assert_called_once_with(schema, None, None, None, 5)


def test_build_and_validate_config_dispatch_image(orchestrator):
    schema = ImageSpec()
    with patch.object(orchestrator, "_handle_image") as mock_hi:
        orchestrator._build_and_validate_config(schema, None, None, None, 5)
        mock_hi.assert_called_once_with(None, None, 5)


def test_build_and_validate_config_image_subschema_raises(orchestrator):
    with pytest.raises(FedbiomedError, match="Subschema selection is not applicable"):
        orchestrator._build_and_validate_config(ImageSpec(), ["sub"], None, None, 5)


def test_build_and_validate_config_unsupported_type(orchestrator):
    with pytest.raises(FedbiomedError, match="Unsupported schema type"):
        orchestrator._build_and_validate_config(42, None, None, None, 5)


# ── _create_accumulator ──────────────────────────────────────────────────────


def test_create_accumulator_invalid_config(orchestrator):
    with pytest.raises(FedbiomedError, match="Invalid accumulator configuration"):
        orchestrator._create_accumulator("not_a_dict")

    with pytest.raises(FedbiomedError, match="Invalid accumulator configuration"):
        orchestrator._create_accumulator({"no_type_key": 1})


def test_create_accumulator_dict(orchestrator):
    acc = orchestrator._create_accumulator({"type": "dict", "children": {}})
    assert isinstance(acc, DictAccumulator)


def test_create_accumulator_sequence(orchestrator):
    acc = orchestrator._create_accumulator(
        {"type": "sequence", "children": [], "indices": []}
    )
    assert isinstance(acc, SequenceAccumulator)


def test_create_accumulator_row(orchestrator):
    acc = orchestrator._create_accumulator(
        {"type": DatasetElementType.ROW, "conf": {"c1": {}}, "columns": ["c1"]}
    )
    assert isinstance(acc, RowAccumulator)


def test_create_accumulator_image(orchestrator):
    acc = orchestrator._create_accumulator(
        {"type": DatasetElementType.IMAGE, "stats": {"mean": {"buffer_size": 10}}}
    )
    assert isinstance(acc, ImageAccumulator)


def test_create_accumulator_unsupported_type(orchestrator):
    with pytest.raises(FedbiomedError, match="Unsupported accumulator type"):
        orchestrator._create_accumulator({"type": "unsupported"})


def test_create_accumulator_dict_recursion(orchestrator):
    """Children are recursively built."""
    config = {
        "type": "dict",
        "children": {"a": {"type": "sequence", "children": [], "indices": []}},
    }
    acc = orchestrator._create_accumulator(config)
    assert isinstance(acc, DictAccumulator)
    assert isinstance(acc.children["a"], SequenceAccumulator)


# ── _handle_dict ─────────────────────────────────────────────────────────────


def test_handle_dict_basic(orchestrator):
    schema = {"a": ImageSpec(), "b": ImageSpec()}
    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        orchestrator._handle_dict(
            schema, subschema=None, stats=None, stats_args=None, n_samples=5
        )
        assert mock_bvc.call_count == 2


def test_handle_dict_errors(orchestrator):
    schema = {"a": 1}

    with pytest.raises(FedbiomedError, match="Args for dict schema must be a dict"):
        orchestrator._handle_dict(schema, None, None, stats_args=[], n_samples=5)

    with pytest.raises(FedbiomedError, match="Args keys .* not found"):
        orchestrator._handle_dict(schema, None, None, stats_args={"z": 1}, n_samples=5)

    with pytest.raises(FedbiomedError, match="Invalid key in subschema"):
        orchestrator._handle_dict(schema, ["z"], None, None, n_samples=5)

    with pytest.raises(
        FedbiomedError, match="Dict item in subschema list must have exactly one key"
    ):
        orchestrator._handle_dict(schema, [{"a": 1, "b": 2}], None, None, n_samples=5)

    with pytest.raises(FedbiomedError, match="Invalid element type"):
        orchestrator._handle_dict(schema, [123], None, None, n_samples=5)

    with pytest.raises(FedbiomedError, match="Duplicate key"):
        orchestrator._handle_dict(schema, ["a", {"a": 1}], None, None, n_samples=5)

    with pytest.raises(
        FedbiomedError, match="Subschema for dict must be list/tuple or dict"
    ):
        orchestrator._handle_dict(schema, "invalid", None, None, n_samples=5)


def test_handle_dict_subschema_as_dict(orchestrator):
    """dict subschema selects a subset of keys."""
    spec_a, spec_b, spec_c = (
        RowSpec(columns=["a"]),
        RowSpec(columns=["b"]),
        RowSpec(columns=["c"]),
    )
    schema = {"a": spec_a, "b": spec_b, "c": spec_c}
    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        orchestrator._handle_dict(
            schema,
            subschema={"a": None, "c": None},
            stats=None,
            stats_args=None,
            n_samples=5,
        )
        assert mock_bvc.call_count == 2
        called_first_args = [call.args[0] for call in mock_bvc.call_args_list]
        assert spec_b not in called_first_args
        assert spec_a in called_first_args
        assert spec_c in called_first_args


def test_handle_dict_subschema_as_dict_invalid_keys(orchestrator):
    schema = {"a": ImageSpec()}
    with pytest.raises(FedbiomedError, match="Invalid keys in subschema"):
        orchestrator._handle_dict(
            schema, subschema={"z": None}, stats=None, stats_args=None, n_samples=5
        )


def test_handle_dict_subschema_list_with_nested_dict_key(orchestrator):
    """List subschema with a single-key dict item resolves correctly."""
    schema = {"a": ImageSpec(), "b": ImageSpec()}
    with patch.object(orchestrator, "_build_and_validate_config"):
        result = orchestrator._handle_dict(
            schema,
            subschema=[{"a": "child_sub"}],
            stats=None,
            stats_args=None,
            n_samples=5,
        )
    assert result["type"] == "dict"
    assert "a" in result["children"]
    assert "b" not in result["children"]


def test_handle_dict_subschema_list_with_string_item(orchestrator):
    """List subschema with a plain string item selects that key."""
    schema = {"a": ImageSpec(), "b": ImageSpec()}
    with patch.object(orchestrator, "_build_and_validate_config"):
        result = orchestrator._handle_dict(
            schema, subschema=["a"], stats=None, stats_args=None, n_samples=5
        )
    assert result["type"] == "dict"
    assert "a" in result["children"]
    assert "b" not in result["children"]


# ── _handle_sequence ─────────────────────────────────────────────────────────


def test_handle_sequence(orchestrator):
    schema = [ImageSpec(), ImageSpec()]
    with patch.object(orchestrator, "_build_and_validate_config"):
        result = orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args=None, n_samples=5
        )
    assert result["type"] == "sequence"
    assert len(result["children"]) == len(schema)
    assert result["indices"] == [0, 1]


def test_handle_sequence_single_element(orchestrator):
    """A single-element sequence returns a sequence config with one child."""
    schema = (ImageSpec(),)
    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        mock_bvc.return_value = {"type": "mock_child"}
        result = orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args=None, n_samples=5
        )
    assert result["type"] == "sequence"
    assert len(result["children"]) == 1
    assert result["children"][0] == {"type": "mock_child"}
    assert result["indices"] == [0]


def test_handle_sequence_subschema_errors(orchestrator):
    schema = [1, 2]

    with pytest.raises(FedbiomedError, match="Subschema for sequence must be list"):
        orchestrator._handle_sequence(
            schema, subschema="wrong", stats=None, stats_args=None, n_samples=5
        )

    with pytest.raises(FedbiomedError, match="does not match schema elements"):
        orchestrator._handle_sequence(
            schema, subschema=[1], stats=None, stats_args=None, n_samples=5
        )


def test_handle_sequence_args_errors(orchestrator):
    schema = [1, 2]

    with pytest.raises(FedbiomedError, match="Args for sequence must be list/tuple"):
        orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args="wrong", n_samples=5
        )

    with pytest.raises(FedbiomedError, match="Args length mismatch"):
        orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args=[1], n_samples=5
        )


def test_handle_sequence_explicit_none_skips_position(orchestrator):
    """Explicit None at a subschema position skips that element from the recursive call."""
    schema = [RowSpec(columns=["c1"]), RowSpec(columns=["c2"])]
    sub_for_first = ["c1"]

    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        mock_bvc.return_value = {
            "type": DatasetElementType.ROW,
            "conf": {},
            "columns": [],
        }
        result = orchestrator._handle_sequence(
            schema,
            subschema=[sub_for_first, None],
            stats=None,
            stats_args=None,
            n_samples=5,
        )

    # Only first position should trigger a recursive call
    assert mock_bvc.call_count == 1
    assert mock_bvc.call_args[0][1] == sub_for_first

    # Second position excluded: one child remains, returned as a sequence config
    assert result["type"] == "sequence"
    assert len(result["children"]) == 1
    assert result["indices"] == [0]


def test_handle_sequence_none_subschema_processes_all(orchestrator):
    """Top-level subschema=None processes all positions — no skipping occurs."""
    schema = [RowSpec(columns=["c1"]), RowSpec(columns=["c2"])]

    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        mock_bvc.return_value = {
            "type": DatasetElementType.ROW,
            "conf": {},
            "columns": [],
        }
        orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args=None, n_samples=5
        )

    assert mock_bvc.call_count == 2


def test_handle_sequence_stats_args_routed(orchestrator):
    """Per-position stats_args are forwarded to the corresponding child call."""
    schema = [RowSpec(columns=["c1"]), RowSpec(columns=["c2"])]
    stats_args = [{"mean": {}}, {"variance": {}}]

    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        mock_bvc.return_value = {
            "type": DatasetElementType.ROW,
            "conf": {},
            "columns": [],
        }
        orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args=stats_args, n_samples=5
        )

    assert mock_bvc.call_count == 2
    assert mock_bvc.call_args_list[0].args[3] == {"mean": {}}
    assert mock_bvc.call_args_list[1].args[3] == {"variance": {}}


def test_handle_sequence_none_schema_item_skips_position(orchestrator):
    """None at a schema position is skipped, as used by (ImageSpec(), None)."""
    schema = (ImageSpec(), None)

    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        mock_bvc.return_value = {"type": DatasetElementType.IMAGE, "stats": {}}
        result = orchestrator._handle_sequence(
            schema, subschema=None, stats=None, stats_args=None, n_samples=5
        )

    # Only the ImageSpec position triggers a recursive call
    assert mock_bvc.call_count == 1

    # None schema item skipped: one child remains, returned as a sequence config
    assert result["type"] == "sequence"
    assert len(result["children"]) == 1
    assert result["indices"] == [0]


def test_handle_sequence_all_none_schema_raises(orchestrator):
    """A schema where every item is None raises an error."""
    with pytest.raises(FedbiomedError, match="no selectable elements"):
        orchestrator._handle_sequence(
            [None, None], subschema=None, stats=None, stats_args=None, n_samples=5
        )


def test_handle_sequence_all_excluded_subschema_raises(orchestrator):
    """A subschema that excludes all positions raises an error."""
    schema = [ImageSpec(), ImageSpec()]
    with pytest.raises(FedbiomedError, match="no selectable elements"):
        orchestrator._handle_sequence(
            schema, subschema=[None, None], stats=None, stats_args=None, n_samples=5
        )


# ── _handle_row ──────────────────────────────────────────────────────────────


@patch(
    "fedbiomed.common.analytics._orchestrator.AnalyticsOrchestrator._compile_leaf_stats"
)
def test_handle_row(mock_compile, orchestrator):
    schema = RowSpec(columns=["c1", "c2"])

    orchestrator._handle_row(schema, None, None, None, n_samples=5)
    assert mock_compile.call_count == 2  # c1 and c2
    mock_compile.reset_mock()

    orchestrator._handle_row(schema, ["c1"], None, None, n_samples=5)
    assert mock_compile.call_count == 1

    with pytest.raises(FedbiomedError, match="Invalid columns in subschema"):
        orchestrator._handle_row(schema, ["z"], None, None, n_samples=5)


@patch(
    "fedbiomed.common.analytics._orchestrator.AnalyticsOrchestrator._compile_leaf_stats"
)
def test_handle_row_return_value(mock_compile, orchestrator):
    """Returned dict has the correct structure with type, conf, and ordered columns."""
    mock_compile.return_value = {"mean": {}}
    schema = RowSpec(columns=["c1", "c2"])

    result = orchestrator._handle_row(schema, None, None, None, n_samples=5)

    assert result["type"] == DatasetElementType.ROW
    assert result["columns"] == ["c1", "c2"]
    assert set(result["conf"].keys()) == {"c1", "c2"}


def test_handle_row_validation_errors(orchestrator):
    schema = RowSpec(columns=["c1"])

    with pytest.raises(FedbiomedError, match="Subschema for ROW must be a list"):
        orchestrator._handle_row(schema, "c1", None, None, n_samples=5)

    with pytest.raises(FedbiomedError, match="Args for ROW must be a dict"):
        orchestrator._handle_row(
            schema, None, None, stats_args=["not", "a", "dict"], n_samples=5
        )

    with pytest.raises(FedbiomedError, match="Invalid columns in args"):
        orchestrator._handle_row(schema, None, None, stats_args={"z": {}}, n_samples=5)


# ── _handle_image ────────────────────────────────────────────────────────────


@patch(
    "fedbiomed.common.analytics._orchestrator.AnalyticsOrchestrator._compile_leaf_stats"
)
def test_handle_image(mock_compile, orchestrator):
    mock_compile.return_value = {"mean": {}}

    # stats path
    result = orchestrator._handle_image(stats=["mean"], stats_args=None, n_samples=5)
    assert result["type"] == DatasetElementType.IMAGE
    assert "stats" in result
    mock_compile.assert_called_once_with(DatasetElementType.IMAGE, ["mean"], None, 5)
    mock_compile.reset_mock()

    # args path
    args = {"histogram": {"bins": 10}}
    mock_compile.return_value = args
    orchestrator._handle_image(stats=None, stats_args=args, n_samples=5)
    mock_compile.assert_called_once_with(DatasetElementType.IMAGE, None, args, 5)


def test_handle_image_args_not_dict_raises(orchestrator):
    with pytest.raises(FedbiomedError, match="Args for IMAGE must be a dict"):
        orchestrator._handle_image(stats=None, stats_args=["wrong"], n_samples=5)


# ── _validate_leaf_stat ──────────────────────────────────────────────────────


@pytest.mark.parametrize("is_explicit", [True, False])
@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_validate_leaf_stat_incompatible(mock_registry, orchestrator, is_explicit):
    mock_registry.check_stat_compatibility.return_value = False
    if is_explicit:
        with pytest.raises(FedbiomedError, match="is not valid for type"):
            orchestrator._validate_leaf_stat(
                DatasetElementType.ROW, "bad_stat", {}, is_explicit=True
            )
    else:
        assert (
            orchestrator._validate_leaf_stat(
                DatasetElementType.ROW, "bad_stat", {}, is_explicit=False
            )
            is False
        )


@pytest.mark.parametrize("is_explicit", [True, False])
@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_validate_leaf_stat_invalid_args(mock_registry, orchestrator, is_explicit):
    # Invalid args always raise, regardless of whether the stat came from stats_args or stats list.
    mock_registry.check_stat_compatibility.return_value = True
    mock_registry.validate_args.side_effect = FedbiomedError("bad args")
    with pytest.raises(FedbiomedError, match="Invalid arguments for statistic"):
        orchestrator._validate_leaf_stat(
            DatasetElementType.ROW, "mean", {}, is_explicit=is_explicit
        )


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_validate_leaf_stat_valid(mock_registry, orchestrator):
    mock_registry.check_stat_compatibility.return_value = True
    mock_registry.validate_args.return_value = None
    assert (
        orchestrator._validate_leaf_stat(
            DatasetElementType.ROW, "mean", {}, is_explicit=True
        )
        is True
    )


# ── _compile_leaf_stats ──────────────────────────────────────────────────────


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_compile_leaf_stats(mock_registry, orchestrator):
    mock_registry.check_stat_compatibility.return_value = True
    mock_registry.validate_args.return_value = True
    mock_registry.get_dependencies.return_value = set()
    mock_registry.get_roots.side_effect = lambda stats, et: set(stats)
    mock_registry.get.return_value = None

    # Default stats
    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=["mean"], stats_args={}, n_samples=10
    )
    assert "mean" in config

    # TODO: Re-enable once stats are approved and the temporary protection is removed

    # # Invalid explicit stat raises
    # mock_registry.check_stat_compatibility.side_effect = lambda s, t: s != "invalid"
    # with pytest.raises(FedbiomedError, match="is not valid for type"):
    #     orchestrator._compile_leaf_stats(
    #         DatasetElementType.ROW, stats=[], stats_args={"invalid": {}}, n_samples=10
    #     )

    # # Invalid default stat is silently skipped
    # mock_registry.check_stat_compatibility.side_effect = lambda s, t: s != "bad_default"
    # config = orchestrator._compile_leaf_stats(
    #     DatasetElementType.ROW, stats=["bad_default"], stats_args={}, n_samples=10
    # )
    # assert "bad_default" not in config


def test_compile_leaf_stats_histogram_without_bin_edges_raises(orchestrator):
    """Requesting histogram raises an error (either not-yet-available or missing bin_edges)."""
    with pytest.raises(FedbiomedError, match="histogram"):
        orchestrator._compile_leaf_stats(
            DatasetElementType.ROW, stats=["histogram"], stats_args={}, n_samples=10
        )


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_compile_leaf_stats_buffer_injection(mock_registry, orchestrator):
    """Stats with uses_buffer=True get buffer_size injected."""
    mock_registry.check_stat_compatibility.return_value = True
    mock_registry.validate_args.return_value = None
    mock_registry.get_dependencies.return_value = set()
    mock_registry.get_roots.side_effect = lambda stats, et: set(stats)

    stat_cfg = MagicMock()
    stat_cfg.uses_buffer = True
    mock_registry.get.return_value = {DatasetElementType.ROW: stat_cfg}

    # using 'mean' because it passes the allowed-stats check
    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=["mean"], stats_args={}, n_samples=42
    )
    assert config["mean"].get("buffer_size") == 42


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_dependencies(mock_registry, orchestrator):
    mock_registry.get_dependencies.side_effect = (
        lambda s, et: {"sum", "count"} if s == "mean" else set()
    )
    mock_registry.check_stat_compatibility.return_value = True
    mock_registry.validate_args.return_value = None
    mock_registry.get_roots.return_value = {"mean"}
    mock_registry.get.return_value = None

    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=["mean"], stats_args={}, n_samples=10
    )

    assert "mean" in config
    assert "sum" not in config
    assert "count" not in config


# ── _resolve_and_validate_roots ──────────────────────────────────────────────


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_resolve_and_validate_roots_conflicting_stat_args(mock_registry, orchestrator):
    """Requesting the same stat twice with different args raises."""
    mock_registry.get_dependencies.side_effect = (
        lambda s, et: {"mean"} if s == "std" else set()
    )
    mock_registry.validate_args.return_value = None
    mock_registry.get_roots.side_effect = lambda stats, et: set(stats)

    with pytest.raises(FedbiomedError, match="Conflicting arguments"):
        orchestrator._resolve_and_validate_roots(
            DatasetElementType.ROW, {"mean": {}, "std": {"ddof": 1}}
        )


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_resolve_and_validate_roots_dependency_conflict(mock_registry, orchestrator):
    """Two stats imply the same dependency but with conflicting arguments."""
    # stat1 depends on dep, stat2 depends on dep
    mock_registry.get_dependencies.side_effect = lambda s, et: {"dep"}
    mock_registry.validate_args.return_value = None

    with pytest.raises(FedbiomedError, match="Conflicting arguments for dependency"):
        orchestrator._resolve_and_validate_roots(
            DatasetElementType.ROW, {"stat1": {"a": 1}, "stat2": {"a": 2}}
        )


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_resolve_and_validate_roots_dep_invalid_args(mock_registry, orchestrator):
    """Dependency that doesn't accept the parent's args raises."""
    mock_registry.get_dependencies.side_effect = (
        lambda s, et: {"sum"} if s == "mean" else set()
    )
    mock_registry.validate_args.side_effect = lambda stat, et, args: (
        (_ for _ in ()).throw(FedbiomedError("bad")) if stat == "sum" else None
    )
    mock_registry.get_roots.return_value = {"mean"}

    with pytest.raises(FedbiomedError, match="implies dependency"):
        orchestrator._resolve_and_validate_roots(
            DatasetElementType.ROW, {"mean": {"bins": 10}}
        )


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_resolve_and_validate_roots_returns_only_roots(mock_registry, orchestrator):
    """Only root stats appear in the final config, not dependencies."""
    mock_registry.get_dependencies.return_value = set()
    mock_registry.validate_args.return_value = None
    mock_registry.get_roots.return_value = {"mean"}

    result = orchestrator._resolve_and_validate_roots(
        DatasetElementType.ROW, {"mean": {}, "sum": {}}
    )
    assert "mean" in result
    assert "sum" not in result


@patch("fedbiomed.common.analytics._orchestrator.AnalyticsRegistry")
def test_resolve_and_validate_roots_nonconflicting_reencounter(
    mock_registry, orchestrator
):
    """A stat already recorded as a dependency is silently accepted when args match."""
    mock_registry.get_dependencies.side_effect = (
        lambda s, et: {"a"} if s == "b" else set()
    )
    mock_registry.validate_args.return_value = None
    mock_registry.get_roots.return_value = {"b"}

    # "b" depends on "a"; both requested with matching args — no conflict
    result = orchestrator._resolve_and_validate_roots(
        DatasetElementType.ROW, {"b": {}, "a": {}}
    )
    assert "b" in result
