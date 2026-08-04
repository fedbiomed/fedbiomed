# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for flatten_fa_output / unflatten_fa_output."""

import numpy as np
import pytest

from fedbiomed.common.utils import flatten_fa_output, unflatten_fa_output

# ---------------------------------------------------------------------------
# flatten_fa_output
# ---------------------------------------------------------------------------


def test_flatten_empty_dict():
    flat, schema = flatten_fa_output({})
    assert flat == []
    assert schema == []


def test_flatten_single_scalar():
    flat, schema = flatten_fa_output({"sum": 100.0})
    assert flat == [100.0]
    assert schema == [["sum"]]


def test_flatten_int_converted_to_float():
    flat, schema = flatten_fa_output({"count": 10})
    assert flat == [10.0]
    assert isinstance(flat[0], float)


def test_flatten_bool_skipped():
    # booleans are a subclass of int but must not be flattened
    flat, schema = flatten_fa_output({"flag": True, "sum": 2.0})
    assert flat == [2.0]
    assert schema == [["sum"]]


def test_flatten_string_skipped():
    flat, schema = flatten_fa_output({"label": "hello", "sum": 1.0})
    assert flat == [1.0]
    assert schema == [["sum"]]


def test_flatten_none_skipped():
    flat, schema = flatten_fa_output({"missing": None, "count": 3})
    assert flat == [3.0]
    assert schema == [["count"]]


def test_flatten_stat_leaf():
    """Typical per-feature stat leaf: keys are emitted in sorted order.

    Dict keys are flattened sorted (count < sum < sum_sq_centered) regardless of
    insertion order, so the layout is deterministic across nodes for secagg.
    """
    output = {"age": {"sum": 450.0, "count": 10, "sum_sq_centered": 20450.0}}
    flat, schema = flatten_fa_output(output)
    assert flat == [10.0, 450.0, 20450.0]
    assert schema == [["age", "count"], ["age", "sum"], ["age", "sum_sq_centered"]]


def test_flatten_is_deterministic_across_insertion_orders():
    """Two nodes with different dict insertion orders flatten identically.

    Regression test: secure aggregation sums encrypted vectors position-by-position
    (masks cancel by index, not by value). If nodes ordered their flat vectors
    differently, the researcher would add e.g. node-1's ``sum`` to node-2's
    ``count`` and silently corrupt the aggregate. Sorted flattening prevents this.
    """
    node1 = {"year": {"sum": 21516413.0, "count": 10667}}
    node2 = {"year": {"count": 17965, "sum": 36233008.0}}  # different insertion order

    flat1, schema1 = flatten_fa_output(node1)
    flat2, schema2 = flatten_fa_output(node2)

    assert schema1 == schema2  # identical layout → positions align across nodes
    # Positional aggregate (as secagg does) now sums matching statistics.
    aggregated = [a + b for a, b in zip(flat1, flat2, strict=True)]
    result = unflatten_fa_output(aggregated, schema1)
    assert result["year"] == {"count": 28632.0, "sum": 57749421.0}


def test_flatten_multiple_features():
    output = {
        "age": {"sum": 450.0, "count": 10},
        "bmi": {"sum": 220.0, "count": 10},
    }
    flat, schema = flatten_fa_output(output)
    assert len(flat) == 4
    assert ["age", "sum"] in schema
    assert ["bmi", "count"] in schema


def test_flatten_deeply_nested():
    output = {"vitals": {"age": {"sum": 100.0}}}
    flat, schema = flatten_fa_output(output)
    assert flat == [100.0]
    assert schema == [["vitals", "age", "sum"]]


def test_flatten_list_values():
    """Histogram-style value: bin_edges and counts are lists."""
    output = {"age": {"histogram": {"bin_edges": [0.0, 1.0, 2.0], "counts": [5, 7]}}}
    flat, schema = flatten_fa_output(output)
    assert flat == [0.0, 1.0, 2.0, 5.0, 7.0]
    assert schema == [
        ["age", "histogram", "bin_edges", 0],
        ["age", "histogram", "bin_edges", 1],
        ["age", "histogram", "bin_edges", 2],
        ["age", "histogram", "counts", 0],
        ["age", "histogram", "counts", 1],
    ]


def test_flatten_tuple_values():
    """Tuples are treated like lists (iterable containers)."""
    output = {"vals": (1.0, 2.0)}
    flat, schema = flatten_fa_output(output)
    assert flat == [1.0, 2.0]
    assert schema == [["vals", 0], ["vals", 1]]


# ---------------------------------------------------------------------------
# numpy scalar types (produced by accumulators: np.float32, np.int32, etc.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.float32(4.5), pytest.approx(4.5)),
        (np.float64(4.5), pytest.approx(4.5)),
        (np.int32(7), 7.0),
        (np.int64(7), 7.0),
    ],
)
def test_flatten_np_scalar_to_float(value, expected):
    flat, _ = flatten_fa_output({"x": value})
    assert len(flat) == 1
    assert isinstance(flat[0], float)
    assert flat[0] == expected


def test_flatten_np_bool_skipped():
    flat, schema = flatten_fa_output({"flag": np.bool_(True), "sum": np.float32(2.0)})
    assert len(flat) == 1
    assert flat[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# unflatten_fa_output
# ---------------------------------------------------------------------------


def test_unflatten_empty():
    assert unflatten_fa_output([], []) == {}


def test_unflatten_single_scalar():
    result = unflatten_fa_output([42.0], [["x"]])
    assert result == {"x": 42.0}


def test_unflatten_stat_leaf():
    flat = [450.0, 10.0, 20450.0]
    schema = [["age", "sum"], ["age", "count"], ["age", "sum_sq_centered"]]
    result = unflatten_fa_output(flat, schema)
    assert result == {"age": {"sum": 450.0, "count": 10.0, "sum_sq_centered": 20450.0}}


def test_unflatten_multiple_features():
    flat = [450.0, 10.0, 220.0, 10.0]
    schema = [["age", "sum"], ["age", "count"], ["bmi", "sum"], ["bmi", "count"]]
    result = unflatten_fa_output(flat, schema)
    assert result == {
        "age": {"sum": 450.0, "count": 10.0},
        "bmi": {"sum": 220.0, "count": 10.0},
    }


def test_unflatten_deeply_nested():
    result = unflatten_fa_output([100.0], [["vitals", "age", "sum"]])
    assert result == {"vitals": {"age": {"sum": 100.0}}}


def test_unflatten_list_values():
    flat = [0.0, 1.0, 2.0, 5.0, 7.0]
    schema = [
        ["age", "histogram", "bin_edges", 0],
        ["age", "histogram", "bin_edges", 1],
        ["age", "histogram", "bin_edges", 2],
        ["age", "histogram", "counts", 0],
        ["age", "histogram", "counts", 1],
    ]
    result = unflatten_fa_output(flat, schema)
    assert result == {
        "age": {"histogram": {"bin_edges": [0.0, 1.0, 2.0], "counts": [5.0, 7.0]}}
    }


# ---------------------------------------------------------------------------
# Round-trip: flatten then unflatten must reproduce the original numeric structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        {"sum": 1.0},
        {"age": {"sum": 450.0, "count": 10, "sum_sq_centered": 20450.0}},
        {
            "age": {"sum": 450.0, "count": 10},
            "bmi": {"sum": 220.0, "count": 10},
        },
        {"vitals": {"age": {"sum": 100.0, "count": 5}}},
        {"age": {"histogram": {"bin_edges": [0.0, 1.0, 2.0], "counts": [5, 7]}}},
        {
            "age": {"sum": 450.0, "count": 10},
            "bmi": {"histogram": {"bin_edges": [18.0, 25.0, 30.0], "counts": [3, 7]}},
        },
        # numpy scalar types from accumulators
        {
            "age": {
                "sum": np.float32(450.0),
                "count": np.int32(10),
                "sum_sq_centered": np.float32(20450.0),
            },
        },
    ],
)
def test_round_trip(output):
    flat, schema = flatten_fa_output(output)
    reconstructed = unflatten_fa_output(flat, schema)
    assert reconstructed == _deep_float(output)


def _deep_float(obj):
    """Recursively convert all numeric values to float (flatten always yields float)."""
    if isinstance(obj, dict):
        return {k: _deep_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_float(v) for v in obj]
    if isinstance(obj, (int, float, np.floating, np.integer)) and not isinstance(
        obj, (bool, np.bool_)
    ):
        return float(obj)
    return obj


def test_round_trip_skips_non_numeric():
    """Strings and booleans in the source are absent from the reconstruction."""
    output = {"label": "A", "flag": True, "sum": 2.0}
    flat, schema = flatten_fa_output(output)
    reconstructed = unflatten_fa_output(flat, schema)
    assert reconstructed == {"sum": 2.0}
    assert "label" not in reconstructed
    assert "flag" not in reconstructed
