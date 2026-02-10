from enum import Enum
from typing import Any

import pytest

from fedbiomed.common.analytics.accumulators._base import Accumulator
from fedbiomed.common.analytics.accumulators._registry import AnalyticsRegistry
from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.exceptions import FedbiomedError


class MockAccumulator(Accumulator):
    def update(self, value: Any) -> None:
        pass

    def finalize(self) -> Any:
        return {}


class MockStats(Enum):
    TEST_STAT = "test_stat"
    STAT_NO_DEPS = "stat_no_deps"
    STAT_WITH_DEPS = "stat_with_deps"
    STAT_WITH_ARGS = "stat_with_args"
    STAT_UNEXPECTED_ARGS = "stat_unexpected_args"
    STAT_MISSING_ARGS = "stat_missing_args"
    STAT_ROW_ONLY = "stat_row_only"
    VEC_STAT = "vec_stat"
    VEC_STAT_BAD = "vec_stat_bad"
    DESC_STAT_BAD = "desc_stat_bad"
    SPLIT_STAT = "split_stat"
    PROTECTED_STAT = "protected_stat"
    OVERLAP_STAT = "overlap_stat"
    COMPAT_STAT = "compat_stat"
    STAT_STRINGS = "stat_strings"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    X = "X"
    Y = "Y"
    P1 = "p1"
    P2 = "p2"
    DEP1 = "dep1"


@pytest.fixture(autouse=True)
def mock_stats(monkeypatch):
    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.Stats", MockStats
    )


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state around tests to avoid side effects."""
    # Copy the inner dictionaries so modifications don't affect restoration
    original_registry = {k: v.copy() for k, v in AnalyticsRegistry._registry.items()}
    yield
    # Restore
    AnalyticsRegistry._registry = original_registry


def test_register_and_get():
    # New API: register(name, valid_for, **config)
    AnalyticsRegistry.register(
        name="test_stat",
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    retrieved = AnalyticsRegistry.get("test_stat")
    assert retrieved is not None
    # retrieved is Dict[DatasetElementType, StatConfig]
    assert DatasetElementType.ROW in retrieved
    # We can check the config object
    assert retrieved[DatasetElementType.ROW].dependencies == set()
    assert retrieved[DatasetElementType.ROW].accumulator_class == MockAccumulator


def test_default_dependencies():
    """Test that dependencies defaults to empty set if not provided."""
    name = "stat_no_deps"
    AnalyticsRegistry.register(
        name=name, valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )

    deps = AnalyticsRegistry.get_dependencies(name, DatasetElementType.ROW)
    assert deps == set()


def test_explicit_dependencies():
    """Test that explicit dependencies are respected."""
    name = "stat_with_deps"
    AnalyticsRegistry.register(
        name=name,
        dependencies={"p1", "p2"},
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    deps = AnalyticsRegistry.get_dependencies(name, DatasetElementType.ROW)
    assert deps == {"p1", "p2"}


def test_get_nonexistent():
    assert AnalyticsRegistry.get("nonexistent_stat") is None


def test_validate_args_success():
    name = "stat_with_args"
    AnalyticsRegistry.register(
        name=name,
        required_args={"arg1", "arg2"},
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    # All args provided
    # Now requires element_type
    AnalyticsRegistry.validate_args(
        name, DatasetElementType.ROW, {"arg1": 1, "arg2": 2}
    )


def test_validate_args_unexpected():
    name = "stat_unexpected_args"
    AnalyticsRegistry.register(
        name=name,
        required_args={"arg1"},
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    # Validation should fail because of 'extra' argument
    with pytest.raises(FedbiomedError, match="received unexpected args"):
        AnalyticsRegistry.validate_args(
            name, DatasetElementType.ROW, {"arg1": 1, "extra": 2}
        )


def test_validate_args_missing_required():
    name = "stat_missing_args"
    AnalyticsRegistry.register(
        name=name,
        required_args={"arg1"},
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )
    with pytest.raises(FedbiomedError, match="missing required args"):
        AnalyticsRegistry.validate_args(name, DatasetElementType.ROW, {"other": 1})


def test_validate_args_unknown_stat():
    # Should raise FedbiomedError for unknown stat
    with pytest.raises(FedbiomedError, match="Unknown statistic"):
        AnalyticsRegistry.validate_args("unknown_stat", DatasetElementType.ROW, {})


def test_validate_args_invalid_type():
    name = "stat_row_only"
    AnalyticsRegistry.register(
        name=name, valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )

    with pytest.raises(FedbiomedError, match="not valid for type"):
        AnalyticsRegistry.validate_args(name, DatasetElementType.IMAGE, {})


def test_vectorizable_flag():
    # Valid registration
    AnalyticsRegistry.register(
        name="vec_stat",
        is_vectorizable=True,
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    meta = AnalyticsRegistry.get("vec_stat")
    # meta is Dict[Type, Config]
    assert meta[DatasetElementType.ROW].is_vectorizable is True


def test_validate_vectorizable_flag_logic():
    # Invalid registration for IMAGE should NOT raise, but should be coerced to False
    AnalyticsRegistry.register(
        name="vec_stat_bad",
        is_vectorizable=True,
        valid_for={DatasetElementType.IMAGE},
        accumulator_class=MockAccumulator,
    )

    stat_config = AnalyticsRegistry.get("vec_stat_bad")
    # Should be False because type is IMAGE
    assert stat_config[DatasetElementType.IMAGE].is_vectorizable is False


def test_validate_descriptor_flag_logic():
    # Invalid registration for ROW should NOT raise, but should be coerced to False
    AnalyticsRegistry.register(
        name="desc_stat_bad",
        is_descriptor=True,
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    stat_config = AnalyticsRegistry.get("desc_stat_bad")
    # Should be False because type is ROW
    assert stat_config[DatasetElementType.ROW].is_descriptor is False


def test_independent_registrations():
    # This is the key feature: same name, added independently
    name = "split_stat"

    # 1. Register for ROW
    AnalyticsRegistry.register(
        name=name,
        required_args={"row_arg"},
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    # 2. Register for IMAGE later
    AnalyticsRegistry.register(
        name=name,
        required_args={"img_arg"},
        valid_for={DatasetElementType.IMAGE},
        accumulator_class=MockAccumulator,
    )

    # Verify ROW config
    AnalyticsRegistry.validate_args(name, DatasetElementType.ROW, {"row_arg": 1})

    # Verify IMAGE config
    AnalyticsRegistry.validate_args(name, DatasetElementType.IMAGE, {"img_arg": 1})

    # Verify cross-pollution is gone (ROW shouldn't accept img_arg)
    with pytest.raises(FedbiomedError, match="missing required args"):
        AnalyticsRegistry.validate_args(name, DatasetElementType.ROW, {"img_arg": 1})


def test_overwrite_protection():
    name = "protected_stat"
    AnalyticsRegistry.register(
        name=name, valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )

    with pytest.raises(FedbiomedError, match="already registered"):
        AnalyticsRegistry.register(
            name=name,
            valid_for={DatasetElementType.ROW},
            accumulator_class=MockAccumulator,
        )


def test_conflicting_args_registration_allowed():
    """Test that registering a stat with overlapping required and optional args is now allowed by the registry (but discouraged)."""
    name = "overlap_stat"
    AnalyticsRegistry.register(
        name=name,
        required_args={"common_arg"},
        optional_args={"common_arg", "other_arg"},
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    # It should register successfully
    assert AnalyticsRegistry.get(name) is not None


def test_default_stats_integrity():
    """Validates that the standard statistics (developer configured) follow best practices.

    This replaces runtime checks. It ensures:
    1. No overlapping args.
    2. Flags match types.
    """
    # We access the configured stats via the registry (which is populated by defaults on import)
    # Note: Since tests assume a clean registry, we might need to re-populate if 'clean_registry' fixture wiped it.
    # The fixture restores the STATE from before the test.
    # If the module loaded, defaults are there.

    from fedbiomed.common.analytics.accumulators._registry import _REGISTRY_STATS

    for stat_def in _REGISTRY_STATS:
        name = stat_def["name"]
        valid_for = stat_def["valid_for"]
        if isinstance(valid_for, DatasetElementType):
            valid_for = {valid_for}
        else:
            valid_for = set(valid_for)

        required = set(stat_def.get("required_args", []))
        optional = set(stat_def.get("optional_args", []))
        is_vec = stat_def.get("is_vectorizable", False)
        is_desc = stat_def.get("is_descriptor", False)

        # 1. Overlap Check
        intersection = required.intersection(optional)
        assert not intersection, f"Stat '{name}' has overlapping args: {intersection}"

        # 2. Vectorizable Check
        if is_vec:
            assert DatasetElementType.ROW in valid_for, (
                f"In Stat '{name}', 'is_vectorizable' is valid for ROW"
            )

        # 3. Descriptor Check
        if is_desc:
            assert DatasetElementType.IMAGE in valid_for, (
                f"In Stat '{name}', 'is_descriptor' is valid for IMAGE"
            )


def test_check_stat_compatibility():
    """Test that check_stat_compatibility returns True if valid, False if invalid type, and raises error if unknown."""
    name = "compat_stat"
    AnalyticsRegistry.register(
        name=name, valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )

    # Valid
    assert (
        AnalyticsRegistry.check_stat_compatibility(name, DatasetElementType.ROW) is True
    )

    # Invalid type
    assert (
        AnalyticsRegistry.check_stat_compatibility(name, DatasetElementType.IMAGE)
        is False
    )

    # Unknown stat
    with pytest.raises(FedbiomedError, match="Unknown statistic"):
        AnalyticsRegistry.check_stat_compatibility("unknown", DatasetElementType.ROW)


def test_get_dependencies_recursive():
    """Test that get_dependencies resolves dependencies recursively."""
    # A -> B -> C
    AnalyticsRegistry.register(
        name="C", valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )
    AnalyticsRegistry.register(
        name="B",
        dependencies="C",
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )
    AnalyticsRegistry.register(
        name="A",
        dependencies="B",
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    deps = AnalyticsRegistry.get_dependencies("A", DatasetElementType.ROW)
    assert deps == {"B", "C"}

    # Test with multiple stats
    # X -> Y
    AnalyticsRegistry.register(
        name="Y", valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )
    AnalyticsRegistry.register(
        name="X",
        dependencies="Y",
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    deps_multi = AnalyticsRegistry.get_dependencies(["A", "X"], DatasetElementType.ROW)
    assert deps_multi == {"B", "C", "Y"}


def test_get_roots():
    """Test get_roots identifies stats that are not implicit dependencies of others in the request."""
    # A -> B -> C
    AnalyticsRegistry.register(
        name="C", valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )
    AnalyticsRegistry.register(
        name="B",
        dependencies="C",
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )
    AnalyticsRegistry.register(
        name="A",
        dependencies="B",
        valid_for={DatasetElementType.ROW},
        accumulator_class=MockAccumulator,
    )

    # 1. Requesting A, B, C.
    # B is needed by A. C is needed by B (and A).
    # Root is A.
    roots = AnalyticsRegistry.get_roots(["A", "B", "C"], DatasetElementType.ROW)
    assert roots == {"A"}

    # 2. Requesting B, C.
    # C is needed by B.
    # Root is B.
    roots = AnalyticsRegistry.get_roots(["B", "C"], DatasetElementType.ROW)
    assert roots == {"B"}

    # 3. Requesting A and an independent D
    AnalyticsRegistry.register(
        name="D", valid_for={DatasetElementType.ROW}, accumulator_class=MockAccumulator
    )
    roots = AnalyticsRegistry.get_roots(["A", "D"], DatasetElementType.ROW)
    assert roots == {"A", "D"}


def test_register_normalization_single_string():
    """Test that single string arguments for dependencies/args are correctly normalized to sets."""
    name = "stat_strings"
    AnalyticsRegistry.register(
        name=name,
        dependencies="dep1",
        required_args="arg1",
        optional_args="opt1",
        valid_for=DatasetElementType.ROW,
        accumulator_class=MockAccumulator,
    )

    config = AnalyticsRegistry.get(name)[DatasetElementType.ROW]
    assert config.dependencies == {"dep1"}
    assert config.required_args == {"arg1"}
    assert config.optional_args == {"opt1"}


def test_register_invalid_stat_name():
    with pytest.raises(FedbiomedError, match="not defined in Stats enum"):
        AnalyticsRegistry.register(
            name="INVALID_NAME_NOT_IN_ENUM",
            valid_for={DatasetElementType.ROW},
            accumulator_class=MockAccumulator,
        )
