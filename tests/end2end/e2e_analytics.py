"""
End-to-end tests for federated analytics using tabular (CSV) datasets.
"""

import math
import os
import time

import pytest
from helpers import (
    add_dataset_to_node,
    clear_component_data,
    clear_experiment_data,
    create_multiple_nodes,
    create_researcher,
    generate_controlled_analytics_dataset,
    generate_sklearn_classification_dataset,
    kill_subprocesses,
    start_nodes,
)

from fedbiomed.common.utils import SHARE_DIR
from fedbiomed.researcher.federated_workflows import Experiment


@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session):
    """Set up 2 nodes, each with an ADNI CSV and a synthetic classification CSV."""

    with create_multiple_nodes(port, 2) as nodes:
        node_1, node_2 = nodes

        researcher = create_researcher(port=port)

        # ADNI dataset — named columns, semicolon-delimited
        adni_path = os.path.join(
            SHARE_DIR, "notebooks", "data", "CSV", "pseudo_adni_mod.csv"
        )
        adni_dataset = {
            "name": "Adni dataset",
            "description": "ADNI DATASET",
            "tags": "#adni-analytics",
            "data_type": "csv",
            "path": adni_path,
        }
        add_dataset_to_node(node_1, adni_dataset)
        add_dataset_to_node(node_2, adni_dataset)

        # Synthetic classification CSV — no header, comma-delimited, 20 features + 1 label
        p1, p2, _ = generate_sklearn_classification_dataset()
        cls_dataset = {
            "name": "Classification CSV",
            "description": "Synthetic classification dataset",
            "tags": "#csv-analytics-classification",
            "data_type": "csv",
        }
        add_dataset_to_node(node_1, {**cls_dataset, "path": p1})
        add_dataset_to_node(node_2, {**cls_dataset, "path": p2})

        # Controlled CSV — named columns A and B, known ground-truth statistics
        cp1, cp2 = generate_controlled_analytics_dataset()
        ctrl_dataset = {
            "name": "Controlled analytics CSV",
            "description": "Synthetic dataset with analytically-known statistics",
            "tags": "#csv-analytics-controlled",
            "data_type": "csv",
        }
        add_dataset_to_node(node_1, {**ctrl_dataset, "path": cp1})
        add_dataset_to_node(node_2, {**ctrl_dataset, "path": cp2})

        node_processes, thread = start_nodes([node_1, node_2])
        print("Sleep 10 seconds to give time for nodes to start")
        time.sleep(10)

        yield node_1, node_2, researcher

        try:
            kill_subprocesses(node_processes)
            thread.join()
        finally:
            print("Clearing researcher data")
            clear_component_data(researcher)


@pytest.fixture
def exp_adni():
    exp = Experiment(tags=["#adni-analytics"])
    yield exp
    clear_experiment_data(exp)


@pytest.fixture
def exp_cls():
    exp = Experiment(tags=["#csv-analytics-classification"])
    yield exp
    clear_experiment_data(exp)


def test_01_analytics_mean(exp_adni):
    """Global mean across 2 nodes for the ADNI tabular dataset."""

    result = exp_adni.analytics.mean()

    assert isinstance(result, dict), "mean() should return a dict keyed by column name"
    assert len(result) > 0, "Result should contain at least one column"
    assert "AGE" in result, "ADNI dataset should have an 'AGE' column"
    assert isinstance(result["AGE"], float), "Mean of 'AGE' should be a float"


def test_02_analytics_variance_available_computable_and_mean(exp_adni):
    """Request variance, inspect available/computable stats, then derive mean
    directly from the same FAResult — no new network request required."""

    # Step 1: fetch variance (sends messages to nodes)
    fa_result = exp_adni.analytics.fetch_stats("variance")

    # Step 2: available_stats lists the raw numeric keys present in the result;
    #         fetching "variance" stores the primitives count, sum, sum_sq — not
    #         a pre-computed "variance" key.
    avail = fa_result.available_stats()
    assert all(k in avail for k in ("count", "sum", "sum_sq")), (
        f"Expected variance primitives (count, sum, sum_sq) in available_stats(): {avail}"
    )

    # Step 3: computable_stats derives which higher-level stats can be calculated;
    #         variance data includes sum + count + sum_of_squares, so mean is computable
    computable = fa_result.computable_stats()
    assert "variance" in computable, (
        f"'variance' not in computable_stats(): {computable}"
    )
    assert "mean" in computable, (
        "'mean' should be computable from variance data (requires sum + count); "
        f"computable_stats() returned: {computable}"
    )

    # Step 4: compute mean directly — no new message sent to nodes
    mean_result = fa_result.global_stats("mean")

    assert isinstance(mean_result, dict), (
        "mean result should be a dict keyed by column name"
    )
    assert "AGE" in mean_result, "ADNI dataset should have an 'AGE' column"
    assert isinstance(mean_result["AGE"], float), "Mean of 'AGE' should be a float"


def test_03_analytics_count(exp_adni):
    """Row count is positive for every ADNI column."""

    count_vals = exp_adni.analytics.count()

    assert isinstance(count_vals, dict)
    assert len(count_vals) > 0
    for col, cnt in count_vals.items():
        assert cnt > 0, f"Count for column '{col}' should be positive"


def test_04_analytics_schema_filter(exp_adni):
    """Analytics respects a column-level schema filter (subset of columns)."""

    selected_cols = ["AGE", "MMSE.bl"]
    fa_result = exp_adni.analytics.fetch_stats("mean", dataset_schema=selected_cols)
    global_mean = fa_result.global_stats("mean")

    assert set(selected_cols).issubset(set(global_mean.keys())), (
        f"Expected columns {selected_cols} to be present in {set(global_mean.keys())}"
    )


def test_05_analytics_no_header_csv(exp_cls):
    """Analytics on a no-header CSV uses numeric column indices (0..20)."""

    fa_result = exp_cls.analytics.fetch_stats(["mean", "count"])
    global_stats = fa_result.global_stats()

    # 20 features + 1 label column
    assert len(global_stats) == 21, (
        f"Expected 21 columns (20 features + 1 label), got {len(global_stats)}"
    )
    for col_stats in global_stats.values():
        assert "mean" in col_stats
        assert col_stats["count"] > 0, "Row count per column must be positive"


def test_06_analytics_caching(exp_adni):
    """Repeated identical requests return the same cached FAResult object."""

    result1 = exp_adni.analytics.fetch_stats("mean")
    result2 = exp_adni.analytics.fetch_stats("mean")

    assert result1 is result2, (
        "A repeated fetch_stats call with identical arguments should return "
        "the cached FAResult (same object identity)"
    )


# Controlled dataset: N=200, A=[1..200], B=2A=[2..400]
# variance([1..N], ddof=1) = N(N+1)/12; for N=200 → 200·201/12 = 3350 exactly
_A_MEAN, _A_VAR = 100.5, 3350.0
_B_MEAN, _B_VAR = 201.0, 13400.0  # mean=2·_A_MEAN, variance=4·_A_VAR


@pytest.fixture
def exp_controlled():
    exp = Experiment(tags=["#csv-analytics-controlled"])
    yield exp
    clear_experiment_data(exp)


def test_07_controlled_full_stats(exp_controlled):
    """count, mean, variance, std, and per-node primitives all match analytical values."""
    fa = exp_controlled.analytics.fetch_stats("variance")

    assert exp_controlled.analytics.count() == {"A": 200, "B": 200}

    mean = fa.global_stats("mean")
    assert mean["A"] == pytest.approx(_A_MEAN, rel=1e-4)
    assert mean["B"] == pytest.approx(_B_MEAN, rel=1e-4)

    var = fa.global_stats("variance")
    assert var["A"] == pytest.approx(_A_VAR, rel=1e-4)
    assert var["B"] == pytest.approx(_B_VAR, rel=1e-4)

    std = fa.global_stats("std")
    assert std["A"] == pytest.approx(math.sqrt(_A_VAR), rel=1e-4)
    assert std["B"] == pytest.approx(math.sqrt(_B_VAR), rel=1e-4)
