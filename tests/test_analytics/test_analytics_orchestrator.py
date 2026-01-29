from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.analytics._analytics_orchestrator import AnalyticsOrchestrator
from fedbiomed.common.dataset_types import DatasetElementType, ImageSpec, RowSpec
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def orchestrator():
    return AnalyticsOrchestrator()


@pytest.fixture
def mock_dataset():
    d = MagicMock()
    d.__len__.return_value = 10
    d.__getitem__.return_value = {"a": [1, 2]}
    # Default schema
    d.get_analytics_schema.return_value = {"a": RowSpec(columns=["col1", "col2"])}
    return d


def test_compute_stats_missing_capability(orchestrator):
    dataset = MagicMock()
    del dataset.get_analytics_schema  # Ensure method missing

    # passing dict as dataset which has no get_analytics_schema
    with pytest.raises(FedbiomedError, match="Dataset does not implement"):
        orchestrator.compute_stats(dataset)


@patch("fedbiomed.common.analytics._analytics_orchestrator.create_accumulator")
def test_compute_stats_flow(mock_create_acc, orchestrator, mock_dataset):
    # Setup
    mock_acc = MagicMock()
    mock_create_acc.return_value = mock_acc
    mock_acc.finalize.return_value = "results"

    # Action
    res = orchestrator.compute_stats(mock_dataset)

    # Assertions
    mock_dataset.get_analytics_schema.assert_called_once()
    mock_create_acc.assert_called_once()
    assert mock_acc.update.call_count == 10
    mock_acc.finalize.assert_called_once()
    assert res == "results"


# --- _handle_dict logic (includes the mixed list support) ---


def test_handle_dict_basic(orchestrator):
    schema = {"a": ImageSpec(), "b": ImageSpec()}

    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        orchestrator._handle_dict(schema, subschema=None, stats=None, args=None)
        assert mock_bvc.call_count == 2


def test_handle_dict_errors(orchestrator):
    schema = {"a": 1}

    # Invalid input type
    with pytest.raises(FedbiomedError, match="Args for dict schema must be a dict"):
        orchestrator._handle_dict(schema, None, None, args=[])

    # Args unknown key
    with pytest.raises(FedbiomedError, match="Args keys .* not found"):
        orchestrator._handle_dict(schema, None, None, args={"z": 1})

    # List subschema with invalid string
    with pytest.raises(FedbiomedError, match="Invalid key in subschema"):
        orchestrator._handle_dict(schema, ["z"], None, None)

    # List subschema with dict > 1 key
    with pytest.raises(
        FedbiomedError, match="Dict item in subschema list must have exactly one key"
    ):
        orchestrator._handle_dict(schema, [{"a": 1, "b": 2}], None, None)

    # List subschema with invalid type
    with pytest.raises(FedbiomedError, match="Invalid element type"):
        orchestrator._handle_dict(schema, [123], None, None)

    # List subschema duplicate
    with pytest.raises(FedbiomedError, match="Duplicate key"):
        orchestrator._handle_dict(schema, ["a", {"a": 1}], None, None)


# --- _handle_sequence ---


def test_handle_sequence(orchestrator):
    schema = [ImageSpec(), ImageSpec()]

    with patch.object(orchestrator, "_build_and_validate_config") as mock_bvc:
        orchestrator._handle_sequence(schema, subschema=None, stats=None, args=None)
        assert mock_bvc.call_count == 2


def test_handle_sequence_errors(orchestrator):
    schema = [1, 2]

    with pytest.raises(FedbiomedError, match="Subschema for sequence must be list"):
        orchestrator._handle_sequence(schema, subschema="wrong", stats=None, args=None)

    with pytest.raises(FedbiomedError, match="Subschema length mismatch"):
        orchestrator._handle_sequence(schema, subschema=[1], stats=None, args=None)


# --- _handle_row ---


@patch(
    "fedbiomed.common.analytics._analytics_orchestrator.AnalyticsOrchestrator._compile_leaf_stats"
)
def test_handle_row(mock_compile, orchestrator):
    schema = RowSpec(columns=["c1", "c2"])

    # 1. Select all
    orchestrator._handle_row(schema, None, None, None)
    assert mock_compile.call_count == 2  # c1 and c2
    mock_compile.reset_mock()

    # 2. Subselect
    orchestrator._handle_row(schema, ["c1"], None, None)
    assert mock_compile.call_count == 1

    # 3. Invalid subselect
    with pytest.raises(FedbiomedError, match="Invalid columns in subschema"):
        orchestrator._handle_row(schema, ["z"], None, None)


# --- _compile_leaf_stats logic & Registry integration ---


@patch("fedbiomed.common.analytics._analytics_orchestrator.AnalyticsRegistry")
def test_compile_leaf_stats(mock_registry, orchestrator):
    # Setup registry mock
    mock_registry.is_valid_for_type.return_value = True
    mock_registry.validate_args.return_value = True  # No exception
    mock_registry.get_dependencies.return_value = set()

    # Case 1: Default stats only
    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=["mean"], args={}
    )
    assert "mean" in config

    # Case 2: Args only
    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=[], args={"max": {"some": "arg"}}
    )
    assert "max" in config
    assert config["max"] == {"some": "arg"}

    # Case 3: Invalid stat (explicit in args)
    mock_registry.is_valid_for_type.side_effect = lambda s, t: s != "invalid"
    with pytest.raises(FedbiomedError, match="is not valid for type"):
        orchestrator._compile_leaf_stats(
            DatasetElementType.ROW, stats=[], args={"invalid": {}}
        )

    # Case 4: Invalid stat (in default list) - should be skipped silently
    mock_registry.is_valid_for_type.side_effect = lambda s, t: s != "bad_default"
    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=["bad_default"], args={}
    )
    assert "bad_default" not in config


@patch("fedbiomed.common.analytics._analytics_orchestrator.AnalyticsRegistry")
def test_dependencies(mock_registry, orchestrator):
    # Setup dependency: 'mean' depends on 'sum' and 'count'
    def get_deps(stat):
        if stat == "mean":
            return {"sum", "count"}
        return set()

    mock_registry.get_dependencies.side_effect = get_deps
    mock_registry.is_valid_for_type.return_value = True

    config = orchestrator._compile_leaf_stats(
        DatasetElementType.ROW, stats=["mean"], args={}
    )

    assert "mean" in config
    assert "sum" in config
    assert "count" in config
