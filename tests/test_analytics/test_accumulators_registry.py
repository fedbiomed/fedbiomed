from typing import Any

import pytest

from fedbiomed.common.analytics.accumulators._base import Accumulator
from fedbiomed.common.analytics.accumulators._registry import AnalyticsRegistry
from fedbiomed.common.constants import Stats
from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.exceptions import FedbiomedError


class MockAccumulator(Accumulator):
    def update(self, value: Any) -> None:
        pass

    def finalize(self) -> Any:
        return {}


class MockAccumulator2(Accumulator):
    def update(self, value: Any) -> None:
        pass

    def finalize(self) -> Any:
        return {}


class MockAccumulatorOneArg(Accumulator):
    def __init__(self, arg1):
        pass

    def update(self, value: Any) -> None:
        pass

    def finalize(self) -> Any:
        return {}


class MockAccumulatorTwoArgs(Accumulator):
    def __init__(self, arg1, arg2):
        pass

    def update(self, value: Any) -> None:
        pass

    def finalize(self) -> Any:
        return {}


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state around tests to avoid side effects."""
    original_registry = {k: v.copy() for k, v in AnalyticsRegistry._REGISTRY.items()}
    yield
    AnalyticsRegistry._REGISTRY = original_registry


def test_validate_args_success():
    AnalyticsRegistry._REGISTRY[DatasetElementType.ROW]["stat_with_args"] = [
        MockAccumulatorTwoArgs
    ]

    AnalyticsRegistry.validate_args(
        "stat_with_args", DatasetElementType.ROW, {"arg1": 1, "arg2": 2}
    )


def test_validate_args_failures():
    AnalyticsRegistry._REGISTRY[DatasetElementType.ROW]["stat_one_arg"] = [
        MockAccumulatorOneArg
    ]

    scenarios = [
        # Unexpected args
        (
            "stat_one_arg",
            DatasetElementType.ROW,
            {"arg1": 1, "extra": 2},
            "received unexpected args",
        ),
        # Missing required args
        ("stat_one_arg", DatasetElementType.ROW, {"other": 1}, "missing required args"),
        # Unknown stat
        ("unknown_stat", DatasetElementType.ROW, {}, "is not valid for type"),
        # Registered for ROW only, checked against IMAGE
        ("stat_one_arg", DatasetElementType.IMAGE, {}, "not valid for type"),
    ]

    for stat, dtype, args, match in scenarios:
        with pytest.raises(FedbiomedError, match=match):
            AnalyticsRegistry.validate_args(stat, dtype, args)


def test_get_accumulators_single_stat():
    AnalyticsRegistry._REGISTRY[DatasetElementType.ROW]["test_stat"] = [MockAccumulator]

    assert AnalyticsRegistry.get_accumulators("test_stat", DatasetElementType.ROW) == [
        MockAccumulator
    ]
    assert (
        AnalyticsRegistry.get_accumulators("test_stat", DatasetElementType.IMAGE) == []
    )
    assert AnalyticsRegistry.get_accumulators("unknown", DatasetElementType.ROW) == []


def test_get_accumulators():
    """get_accumulators returns a unique set of accumulator classes across requested stats."""
    AnalyticsRegistry._REGISTRY[DatasetElementType.ROW]["alias_a"] = [
        MockAccumulator,
        MockAccumulator2,
    ]
    AnalyticsRegistry._REGISTRY[DatasetElementType.ROW]["alias_b"] = [MockAccumulator]

    result = AnalyticsRegistry.get_accumulators(
        ["alias_a", "alias_b"], DatasetElementType.ROW
    )
    assert set(result) == {MockAccumulator, MockAccumulator2}

    result_partial = AnalyticsRegistry.get_accumulators(
        ["alias_a", "unknown"], DatasetElementType.ROW
    )
    assert set(result_partial) == {MockAccumulator, MockAccumulator2}


def test_default_stats_integrity():
    with pytest.raises(FedbiomedError, match="missing required args"):
        AnalyticsRegistry.validate_args("histogram", DatasetElementType.ROW, {})

    AnalyticsRegistry.validate_args("count", DatasetElementType.ROW, {})


def test_registry_keys_are_valid_stats():
    """Every stat name in _REGISTRY is a value defined in the Stats enum."""
    valid_values = {s.value for s in Stats}
    for element_type, stats in AnalyticsRegistry._REGISTRY.items():
        for stat_name in stats:
            assert stat_name in valid_values, (
                f"'{stat_name}' registered under {element_type.name} is not a valid Stats value"
            )
