"""
FedComBat workflow of the tutorial: harmonise the nodes' data, then train and
validate a model on it.
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from experiments.training_plans.fedcombat import FedComBatTrainingPlan
from fedbiomed.common.constants import (
    HARMONIZED_PREFIX,
    NODE_DYNAMIC_DATA_FOLDER,
    PreprocType,
)
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment
from helpers import (
    add_dataset_to_node,
    clear_experiment_data,
    generate_fedcombat_dataset,
    researcher_requests,
    run_node_cli,
)

TAG = "#e2e-fedcombat"
COVARIATES = ["SEX", "AGE", "PTEDUCAT"]
PHENOTYPES = ["CDRSB.bl", "RAVLT.forgetting.bl"]
ROUNDS = 2


@pytest.fixture(scope="module")
def nodes(federation):
    """Two running nodes, each holding one site of the FedComBat data, by node id."""
    with federation.nodes(2) as nodes:
        for node, path in zip(nodes, generate_fedcombat_dataset(), strict=True):
            add_dataset_to_node(
                node,
                {
                    "name": "FedComBat site",
                    "description": "Site data with a site effect to harmonise",
                    "tags": TAG,
                    "data_type": "csv",
                    "path": path,
                },
            )
        federation.start(nodes)
        by_id = {node.get("default", "id"): node for node in nodes}
        researcher_requests(list(by_id))
        yield by_id


@pytest.fixture(scope="module")
def trained(nodes):
    """Experiment trained on harmonised data, with its source and harmonised dataset ids."""
    exp = Experiment(
        tags=[TAG],
        training_plan_class=FedComBatTrainingPlan,
        model_args={"in_features": 4, "out_features": 1},
        training_args={
            "loader_args": {"batch_size": 20},
            "optimizer_args": {"lr": 1e-3},
            "epochs": 2,
            "test_ratio": 0.2,
            "test_metric": MetricTypes.MEAN_SQUARE_ERROR,
            "test_on_global_updates": True,
        },
        round_limit=ROUNDS,
        aggregator=FedAverage(),
        node_selection_strategy=None,
    )
    exp.set_preprocessing(
        PreprocType.FEDCOMBAT, {"covariates": COVARIATES, "phenotypes": PHENOTYPES}
    )
    source_ids = {n: d["dataset_id"] for n, d in exp.training_data().data().items()}

    # Harmonises before the first round
    exp.run()

    yield SimpleNamespace(
        exp=exp,
        source_ids=source_ids,
        ids={n: d["dataset_id"] for n, d in exp.training_data().data().items()},
    )
    clear_experiment_data(exp)


def _site_data(nodes, dataset_ids):
    """Each node's dataset CSV, by node id."""
    data = {}
    for node_id, node in nodes.items():
        dataset_manager = DatasetManager(node.getpath("default", "db"))
        path = dataset_manager.get_dataset(dataset_ids[node_id], scope="both")["path"]
        data[node_id] = pd.read_csv(path)
    return data


def _site_gap(data, column):
    """Distance between the two sites' means of `column`."""
    first, second = (site[column].mean() for site in data.values())
    return abs(first - second)


def test_each_node_works_on_a_new_harmonised_dataset(nodes, trained):
    assert set(trained.ids) == set(nodes)
    assert all(trained.ids[n] != trained.source_ids[n] for n in nodes)
    # Each harmonised dataset derives from its node's source dataset
    assert {
        n: d["parent_dataset_id"] for n, d in trained.exp.training_data().data().items()
    } == trained.source_ids


def test_harmonisation_removes_the_site_effect(nodes, trained):
    source = _site_data(nodes, trained.source_ids)
    harmonised = _site_data(nodes, trained.ids)

    for phenotype in PHENOTYPES:
        # Sites are shifted by 5 in the source data
        assert _site_gap(harmonised, phenotype) < 0.2 * _site_gap(source, phenotype)
    for node_id in nodes:
        for covariate in COVARIATES:
            np.testing.assert_allclose(
                harmonised[node_id][covariate], source[node_id][covariate]
            )


def test_every_round_trains_on_the_harmonised_data(nodes, trained):
    replies = trained.exp.training_replies()

    assert sorted(replies) == list(range(ROUNDS))
    for round_replies in replies.values():
        assert {n: r["dataset_id"] for n, r in round_replies.items()} == trained.ids
        assert all(r["success"] for r in round_replies.values())


def test_every_node_validates_the_model_on_the_harmonised_data(nodes, trained):
    # Private: the monitor has no public accessor to the received metrics
    metrics = trained.exp.monitor()._metric_store

    for node_id in nodes:
        assert metrics[node_id]["testing_global_updates"]


def test_harmonisation_runs_once_for_the_experiment(nodes, trained):
    assert not trained.exp.preprocessing.execute()


def test_harmonised_dataset_is_not_searchable_by_tags(nodes, trained):
    # Not a second `Experiment`: clearing it stops the server `trained` uses
    found = researcher_requests(list(nodes)).search([TAG])

    assert {n: [d["dataset_id"] for d in found[n]] for n in nodes} == {
        n: [trained.source_ids[n]] for n in nodes
    }


def test_temporary_standardised_dataset_is_removed(nodes, trained):
    """Only the harmonised file is left in the node's dynamic data folder."""
    for node in nodes.values():
        files = os.listdir(os.path.join(node.root, NODE_DYNAMIC_DATA_FOLDER))

        assert len(files) == 1
        assert HARMONIZED_PREFIX in files[0]


# Runs last: deletes the datasets the other tests use
def test_deleting_the_source_deletes_its_derived_dataset(nodes, trained):
    node_id, node = next(iter(nodes.items()))

    assert run_node_cli(node, "dataset", "delete", "--all").returncode == 0

    dataset_manager = DatasetManager(node.getpath("default", "db"))
    assert dataset_manager.get_dataset(trained.ids[node_id], scope="dynamic") is None
