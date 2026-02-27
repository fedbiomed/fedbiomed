# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Tests for ImageAccumulator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from fedbiomed.common.analytics.accumulators._image import ImageAccumulator
from fedbiomed.common.constants import FedbiomedError
from fedbiomed.common.dataset_types import DatasetElementType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_registry(monkeypatch):
    """Mocks AnalyticsRegistry.get_accumulator_class to return a controllable class.

    Returns:
        (mock_class, mock_instance): The mock accumulator class and the instance
        it returns on construction.
    """
    mock_instance = MagicMock()
    mock_instance.update = MagicMock()
    mock_instance.finalize = MagicMock(return_value={"result": 1.0})

    mock_class = MagicMock(return_value=mock_instance)

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=mock_class),
    )

    return mock_class, mock_instance


@pytest.fixture
def mock_registry_none(monkeypatch):
    """Mocks AnalyticsRegistry.get_accumulator_class to return None (no class registered)."""
    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=None),
    )


@pytest.fixture
def image_2d():
    return np.ones((64, 64), dtype=np.float32)


@pytest.fixture
def image_chw():
    return np.ones((3, 32, 32), dtype=np.float32)


# =============================================================================
# __init__ tests
# =============================================================================


def test_init_empty_stats():
    """No stats in config → no accumulators instantiated, no error."""
    acc = ImageAccumulator({"type": DatasetElementType.IMAGE, "stats": {}})
    assert acc.stats_config == {}
    assert acc.accumulators == {}


def test_init_missing_stats_key():
    """Missing 'stats' key is treated as empty stats (defaults to {})."""
    acc = ImageAccumulator({"type": DatasetElementType.IMAGE})
    assert acc.stats_config == {}
    assert acc.accumulators == {}


def test_init_single_stat_no_args(mock_registry):
    """Single stat with no args instantiates the accumulator class with no kwargs."""
    mock_class, mock_instance = mock_registry

    acc = ImageAccumulator({"stats": {"mean": {}}})

    assert "mean" in acc.accumulators
    mock_class.assert_called_once_with()
    assert acc.accumulators["mean"] is mock_instance


def test_init_single_stat_with_args(mock_registry):
    """Single stat with args passes kwargs to the accumulator class."""
    mock_class, mock_instance = mock_registry

    acc = ImageAccumulator({"stats": {"histogram": {"bin_edges": [0, 1, 2]}}})

    assert "histogram" in acc.accumulators
    mock_class.assert_called_once_with(bin_edges=[0, 1, 2])
    assert acc.accumulators["histogram"] is mock_instance


def test_init_multiple_stats(monkeypatch):
    """Multiple stats each produce a distinct accumulator instance."""
    instances = [MagicMock(), MagicMock()]
    mock_class = MagicMock(side_effect=instances)

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=mock_class),
    )

    acc = ImageAccumulator({"stats": {"mean": {}, "variance": {}}})

    assert set(acc.accumulators.keys()) == {"mean", "variance"}
    assert acc.accumulators["mean"] is instances[0]
    assert acc.accumulators["variance"] is instances[1]
    assert mock_class.call_count == 2


def test_init_unregistered_stat_raises(mock_registry_none):
    """Stat with no registered accumulator class raises FedbiomedError."""
    with pytest.raises(
        FedbiomedError, match="No accumulator registered for stat 'unknown'"
    ):
        ImageAccumulator({"stats": {"unknown": {}}})


def test_init_registry_queried_with_image_type(monkeypatch):
    """Registry is always queried for DatasetElementType.IMAGE, never ROW."""
    mock_get = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        mock_get,
    )

    ImageAccumulator({"stats": {"mean": {}, "count": {}}})

    for c in mock_get.call_args_list:
        assert c.args[1] == DatasetElementType.IMAGE


def test_init_stats_config_stored(mock_registry):
    """stats_config attribute mirrors the 'stats' entry from config."""
    stats = {"mean": {}, "histogram": {"bin_edges": [0.0, 1.0]}}
    acc = ImageAccumulator({"stats": stats})
    assert acc.stats_config == stats


# =============================================================================
# update tests
# =============================================================================


def test_update_no_stats_no_error():
    """update() with no registered stats completes silently."""
    acc = ImageAccumulator({"stats": {}})
    acc.update(np.zeros((3, 64, 64)))  # should not raise


def test_update_passes_full_array_to_accumulator(mock_registry, image_chw):
    """update() passes the full image array to each stat accumulator."""
    _, mock_instance = mock_registry

    acc = ImageAccumulator({"stats": {"mean": {}}})
    acc.update(image_chw)

    mock_instance.update.assert_called_once()
    passed = mock_instance.update.call_args[0][0]
    np.testing.assert_array_equal(passed, image_chw)


def test_update_converts_list_to_ndarray(mock_registry):
    """update() converts a plain Python list to np.ndarray before dispatch."""
    _, mock_instance = mock_registry

    acc = ImageAccumulator({"stats": {"mean": {}}})
    acc.update([[1.0, 2.0], [3.0, 4.0]])

    passed = mock_instance.update.call_args[0][0]
    assert isinstance(passed, np.ndarray)
    np.testing.assert_array_equal(passed, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_update_dispatches_to_all_stats(monkeypatch):
    """update() calls update on every registered stat accumulator."""
    instances = [MagicMock(), MagicMock()]
    mock_class = MagicMock(side_effect=instances)

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=mock_class),
    )

    image = np.ones((3, 8, 8))
    acc = ImageAccumulator({"stats": {"mean": {}, "count": {}}})
    acc.update(image)

    for inst in instances:
        inst.update.assert_called_once()
        passed = inst.update.call_args[0][0]
        np.testing.assert_array_equal(passed, image)


def test_update_same_array_passed_to_all_stats(monkeypatch):
    """All stat accumulators receive the exact same ndarray object on update."""
    received = []

    class CapturingAcc:
        def update(self, v):
            received.append(v)

        def finalize(self):
            return None

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=CapturingAcc),
    )

    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    acc = ImageAccumulator({"stats": {"mean": {}, "variance": {}}})
    acc.update(image)

    assert len(received) == 2
    np.testing.assert_array_equal(received[0], image)
    np.testing.assert_array_equal(received[1], image)


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),  # 2D grayscale
        (3, 64, 64),  # 3D CHW
        (1, 128, 128),  # single-channel CHW
        (4, 16, 16),  # 4-channel
        (10,),  # 1D edge case
    ],
)
def test_update_various_image_shapes(mock_registry, shape):
    """update() does not restrict image shape; any ndarray is accepted."""
    _, mock_instance = mock_registry

    acc = ImageAccumulator({"stats": {"mean": {}}})
    image = np.zeros(shape, dtype=np.float32)
    acc.update(image)  # must not raise

    passed = mock_instance.update.call_args[0][0]
    assert passed.shape == shape


def test_update_multiple_calls_accumulate(mock_registry):
    """Multiple calls to update() each invoke the sub-accumulator's update."""
    _, mock_instance = mock_registry

    acc = ImageAccumulator({"stats": {"mean": {}}})
    for _ in range(5):
        acc.update(np.ones((3, 8, 8)))

    assert mock_instance.update.call_count == 5


# =============================================================================
# finalize tests
# =============================================================================


def test_finalize_empty_stats_returns_empty_dict():
    """finalize() with no stats returns an empty dictionary."""
    acc = ImageAccumulator({"stats": {}})
    assert acc.finalize() == {}


def test_finalize_single_stat(mock_registry):
    """finalize() returns {stat_name: finalized_value} for a single stat."""
    _, mock_instance = mock_registry
    mock_instance.finalize.return_value = 42.0

    acc = ImageAccumulator({"stats": {"mean": {}}})
    result = acc.finalize()

    assert result == {"mean": 42.0}


def test_finalize_multiple_stats(monkeypatch):
    """finalize() collects results from all stats into a single dict."""
    results_map = {"mean": 7.5, "count": 100}
    instances = {
        k: MagicMock(**{"finalize.return_value": v}) for k, v in results_map.items()
    }

    call_order = list(instances.values())
    mock_class = MagicMock(side_effect=call_order)

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=mock_class),
    )

    acc = ImageAccumulator({"stats": {"mean": {}, "count": {}}})
    result = acc.finalize()

    assert set(result.keys()) == {"mean", "count"}
    assert result["mean"] == 7.5
    assert result["count"] == 100


def test_finalize_returns_whatever_sub_accumulator_returns(mock_registry):
    """finalize() is transparent: it forwards the sub-accumulator's return value."""
    _, mock_instance = mock_registry

    complex_result = {"bin_edges": [0.0, 1.0, 2.0], "counts": [3, 5]}
    mock_instance.finalize.return_value = complex_result

    acc = ImageAccumulator({"stats": {"histogram": {"bin_edges": [0.0, 1.0, 2.0]}}})
    result = acc.finalize()

    assert result["histogram"] is complex_result


def test_finalize_calls_each_sub_finalize_once(monkeypatch):
    """finalize() calls finalize() exactly once on each sub-accumulator."""
    instances = [MagicMock(), MagicMock()]
    for inst in instances:
        inst.finalize.return_value = None
    mock_class = MagicMock(side_effect=instances)

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        MagicMock(return_value=mock_class),
    )

    acc = ImageAccumulator({"stats": {"mean": {}, "variance": {}}})
    acc.finalize()

    for inst in instances:
        inst.finalize.assert_called_once()


def test_finalize_after_multiple_updates(mock_registry):
    """finalize() still delegates to sub-accumulators after repeated updates."""
    _, mock_instance = mock_registry
    mock_instance.finalize.return_value = np.array([1.0, 2.0, 3.0])

    acc = ImageAccumulator({"stats": {"mean": {}}})
    for _ in range(3):
        acc.update(np.zeros((3, 8, 8)))

    result = acc.finalize()
    np.testing.assert_array_equal(result["mean"], np.array([1.0, 2.0, 3.0]))
