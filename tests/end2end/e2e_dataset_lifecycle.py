"""
Datasets added and deleted with the node CLI while the node runs, and what the
researcher then finds.
"""

from types import SimpleNamespace

import pytest
from fedbiomed.researcher.federated_workflows import Experiment
from helpers import (
    add_dataset_to_node,
    clear_experiment_data,
    generate_controlled_analytics_dataset,
    researcher_requests,
    run_node_cli,
)

TAG = "#e2e-lifecycle"
# What the node CLI prints for a node without any dataset
NO_DATASET = "No data has been set up."


@pytest.fixture(scope="module")
def nodes(federation):
    """Two running nodes; the first one holds the datasets of each test."""
    with federation.nodes(2) as (node_1, node_2):
        federation.start((node_1, node_2))
        ids = [node.get("default", "id") for node in (node_1, node_2)]
        researcher_requests(ids)
        yield SimpleNamespace(node_1=node_1, node_1_id=ids[0], ids=ids)


@pytest.fixture(scope="module")
def dataset():
    """A 100-row CSV dataset with columns A and B."""
    path, _ = generate_controlled_analytics_dataset()
    return {
        "name": "Lifecycle CSV",
        "description": "Dataset registered by the lifecycle tests",
        "tags": TAG,
        "data_type": "csv",
        "path": path,
    }


@pytest.fixture
def registered(nodes, dataset):
    """The dataset registered on the first node, deleted after the test."""
    add_dataset_to_node(nodes.node_1, dataset)
    yield dataset
    run_node_cli(nodes.node_1, "dataset", "delete", "--all")


def test_registered_dataset_is_found_and_read_by_the_researcher(nodes, registered):
    assert registered["name"] in run_node_cli(nodes.node_1, "dataset", "list").stdout

    exp = Experiment(tags=[TAG])
    try:
        assert exp.training_data().node_ids() == [nodes.node_1_id]
        assert exp.analytics.count() == {"A": 100, "B": 100}
    finally:
        clear_experiment_data(exp)


def test_conflicting_tags_are_refused(nodes, registered):
    result = add_dataset_to_node(
        nodes.node_1,
        {**registered, "name": "Conflicting CSV", "tags": f"{TAG},#extra"},
        check=False,
    )

    assert result.returncode == 1
    assert "FB322" in result.stdout
    assert "Traceback" not in result.stdout
    listed = run_node_cli(nodes.node_1, "dataset", "list").stdout
    assert registered["name"] in listed and "Conflicting CSV" not in listed


def test_dataset_below_the_node_minimum_is_refused(federation, dataset):
    with federation.nodes(
        1, config_sections={"security": {"minimum_samples": "1000"}}
    ) as (strict_node,):
        result = add_dataset_to_node(strict_node, dataset, check=False)

        assert result.returncode == 1
        assert "FB632" in result.stdout
        assert "Traceback" not in result.stdout
        assert NO_DATASET in run_node_cli(strict_node, "dataset", "list").stdout


def test_deleted_dataset_is_no_longer_found(nodes, registered):
    """No node restart is needed."""
    # Option 1: the node's only dataset
    result = run_node_cli(nodes.node_1, "dataset", "delete", stdin="1\n")

    assert result.returncode == 0
    assert NO_DATASET in run_node_cli(nodes.node_1, "dataset", "list").stdout
    assert researcher_requests(nodes.ids).search([TAG]) == {}


def test_delete_all_empties_the_registry(nodes, registered, dataset):
    add_dataset_to_node(
        nodes.node_1, {**dataset, "name": "Other CSV", "tags": "#e2e-lifecycle-other"}
    )

    result = run_node_cli(nodes.node_1, "dataset", "delete", "--all")

    assert result.returncode == 0
    assert NO_DATASET in run_node_cli(nodes.node_1, "dataset", "list").stdout
    assert researcher_requests(nodes.ids).list()[nodes.node_1_id] == []
