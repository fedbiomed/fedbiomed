import pytest

from fedbiomed.common.analytics._analytics_registry import (
    AnalyticsRegistry,
    StatMetadata,
)
from fedbiomed.common.dataset_types import DatasetElementType


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state around tests to avoid side effects."""
    # We can't deepcopy the registry easily if it's just a dict of dataclasses,
    # but we can copy the dict.
    original_registry = AnalyticsRegistry._registry.copy()
    yield
    # Restore
    AnalyticsRegistry._registry = original_registry


def test_register_and_get():
    meta = StatMetadata(name="test_stat", valid_for={DatasetElementType.ROW})
    AnalyticsRegistry.register(meta)

    retrieved = AnalyticsRegistry.get("test_stat")
    assert retrieved == meta
    assert retrieved.name == "test_stat"


def test_get_nonexistent():
    assert AnalyticsRegistry.get("nonexistent_stat") is None


def test_validate_args_success():
    name = "stat_with_args"
    AnalyticsRegistry.register(
        StatMetadata(
            name=name,
            required_args={"arg1", "arg2"},
            valid_for={DatasetElementType.ROW},
        )
    )

    # All args provided
    AnalyticsRegistry.validate_args(name, {"arg1": 1, "arg2": 2, "optional": 3})


def test_validate_args_missing_required():
    name = "stat_missing_args"
    AnalyticsRegistry.register(
        StatMetadata(
            name=name, required_args={"arg1"}, valid_for={DatasetElementType.ROW}
        )
    )

    with pytest.raises(ValueError, match="missing required args"):
        AnalyticsRegistry.validate_args(name, {"other": 1})


def test_validate_args_unknown_stat():
    # Should raise ValueError for unknown stat
    with pytest.raises(ValueError, match="Unknown statistic"):
        AnalyticsRegistry.validate_args("unknown_stat", {})


def test_get_dependencies_simple():
    name = "dependent_stat"
    AnalyticsRegistry.register(
        StatMetadata(
            name=name, dependencies={"dep1"}, valid_for={DatasetElementType.ROW}
        )
    )
    AnalyticsRegistry.register(
        StatMetadata(name="dep1", valid_for={DatasetElementType.ROW})
    )

    deps = AnalyticsRegistry.get_dependencies(name)
    assert deps == {"dep1"}


def test_get_dependencies_recursive():
    # A -> B -> C
    AnalyticsRegistry.register(
        StatMetadata(name="A", dependencies={"B"}, valid_for={DatasetElementType.ROW})
    )
    AnalyticsRegistry.register(
        StatMetadata(name="B", dependencies={"C"}, valid_for={DatasetElementType.ROW})
    )
    AnalyticsRegistry.register(
        StatMetadata(name="C", valid_for={DatasetElementType.ROW})
    )

    deps_a = AnalyticsRegistry.get_dependencies("A")
    assert deps_a == {"B", "C"}

    deps_b = AnalyticsRegistry.get_dependencies("B")
    assert deps_b == {"C"}


def test_get_dependencies_unknown_stat():
    assert AnalyticsRegistry.get_dependencies("unknown") == set()


def test_is_valid_for_type_specific():
    name = "row_only"
    AnalyticsRegistry.register(
        StatMetadata(name=name, valid_for={DatasetElementType.ROW})
    )

    assert AnalyticsRegistry.is_valid_for_type(name, DatasetElementType.ROW) is True
    assert AnalyticsRegistry.is_valid_for_type(name, DatasetElementType.IMAGE) is False


def test_is_valid_for_type_explicit_all():
    # Test strict explicit all (simulating what was previously default)
    name = "all_explict"
    all_types = {
        DatasetElementType.ROW,
        DatasetElementType.IMAGE,
        DatasetElementType.UNKNOWN,
    }
    AnalyticsRegistry.register(StatMetadata(name=name, valid_for=all_types))

    assert AnalyticsRegistry.is_valid_for_type(name, DatasetElementType.ROW) is True
    assert AnalyticsRegistry.is_valid_for_type(name, DatasetElementType.IMAGE) is True


def test_is_valid_for_type_unknown_stat():
    # Should raise ValueError for unknown stats
    with pytest.raises(ValueError, match="Unknown statistic"):
        AnalyticsRegistry.is_valid_for_type("unknown", DatasetElementType.ROW)


def test_default_registrations():
    # Verify some default stats are registered correctly
    mean = AnalyticsRegistry.get("mean")
    assert mean is not None
    assert "count" in mean.dependencies
    assert (
        DatasetElementType.UNKNOWN not in mean.valid_for
    )  # Check NN compatibility default

    hist = AnalyticsRegistry.get("histogram")
    assert hist is not None
    assert "bin_edges" in hist.required_args
    assert DatasetElementType.ROW in hist.valid_for
    assert DatasetElementType.UNKNOWN not in hist.valid_for
    assert DatasetElementType.IMAGE not in hist.valid_for


def test_stat_metadata_defaults():
    meta = StatMetadata(name="minimal", valid_for={DatasetElementType.ROW})
    assert meta.dependencies == set()
    assert meta.required_args == set()
    assert meta.is_descriptor is False
