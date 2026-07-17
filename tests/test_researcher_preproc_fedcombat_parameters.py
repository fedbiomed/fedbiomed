"""Tests for _FedCombatParameters computing parameters for FedComBat."""

import math
from unittest.mock import MagicMock

import pytest
import torch
from pytest import MonkeyPatch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.federated_workflows._federated_workflow import (
    FederatedAnalytics,
)
from fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat_model import (
    _FedCombatTrainModel,
)
from fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat_parameters import (
    _FedCombatParameters,
)
from fedbiomed.researcher.requests import Requests


@pytest.fixture
def mock_fds():
    fds = MagicMock(spec=FederatedDataset)
    fds.data.return_value = {
        "n1": {"dataset_id": "ds1", "data_type": "csv"},
        "n2": {"dataset_id": "ds2", "data_type": "csv"},
    }
    fds.node_ids.return_value = ["n1", "n2"]
    return fds


@pytest.fixture
def mock_reqs():
    return MagicMock(spec=Requests)


@pytest.fixture
def mock_analytics():
    analytics = MagicMock(spec=FederatedAnalytics)

    class MockStatsResult:
        """Simple container mimicking the object returned by FederatedAnalytics.fetch_stats()."""

        def __init__(self, stats_dict):
            self._stats_dict = stats_dict

        def global_stats(self, stat_name):
            # This mimics the aggregation logic used in _FedCombatParameters._compute_global_mean_std
            if stat_name not in self._stats_dict:
                raise KeyError(f"Unknown stat name: {stat_name}")
            return self._stats_dict[stat_name]

    # Configure fetch_stats to return a structure compatible with _compute_global_mean_std
    # Each stat ("mean", "std") returns per-node tensors
    def fetch_stats_side_effect(stat_names, _emit_log):
        result = {
            "mean": {
                "cov1": torch.tensor([1.0]),
                "cov2": torch.tensor([2.0]),
                "phen1": torch.tensor([3.0]),
            },
            "variance": {
                "cov1": torch.tensor([1.0]),
                "cov2": torch.tensor([4.0]),
                "phen1": torch.tensor([9.0]),
            },
            "count": {
                "cov1": torch.tensor([1.0]),
                "cov2": torch.tensor([4.0]),
                "phen1": torch.tensor([9.0]),
            },
        }
        return MockStatsResult(result)

    analytics.fetch_stats.side_effect = fetch_stats_side_effect
    return analytics


@pytest.fixture
def mock_fedcombat_tm():
    fedcombat_tm = MagicMock(spec=_FedCombatTrainModel)
    fedcombat_tm.execute.return_value = (
        # biological_model, global_bias_model, local_bias_models
        "bio_model",
        "global_bias_model",
        {
            "n1": {"local_bias_model": "dummy"},
            "n2": {"local_bias_model": "dummy"},
        },
    )
    return fedcombat_tm


@pytest.fixture
def mock_experiment_class():
    return MagicMock(spec=Experiment)


@pytest.fixture
def fedcombat_params(mock_fds, mock_reqs, tmp_path, mock_experiment_class):
    return _FedCombatParameters(
        experiment_id="exp1",
        researcher_id="r1",
        fds=mock_fds,
        reqs=mock_reqs,
        experimentation_folder=str(tmp_path),
        nodes=["n1", "n2"],
        experiment_class=mock_experiment_class,
    )


def test_fedcombat_parameters_compute(
    mock_fedcombat_tm, mock_analytics, fedcombat_params
):
    # Step 1: compute mean and variance
    step1_params = []
    step1_params_local = {
        "covariates": ["cov1", "cov2"],
        "phenotypes": ["phen1"],
    }
    step1_output_keys = [
        "covariates",
        "phenotypes",
        "global_mean_covariates",
        "global_mean_phenotypes",
        "global_std_covariates",
        "global_std_phenotypes",
    ]
    step1_node_output_keys = []

    # Step 2: train model
    step2_params = []
    step2_params_local = {
        "covariates": ["cov1", "cov2"],
        "phenotypes": ["phen1"],
        "training_args": {"epochs": 5},
        "model_args": {"hidden_size": 10},
        "rounds": 3,
        "dict_std_fds": {"n1": {"dataset_id": "ds1"}, "n2": {"dataset_id": "ds2"}},
    }
    step2_output_keys = [
        "biological_model",
        "global_bias_model",
    ]
    step2_node_output_keys = ["local_bias_model"]

    # Step 3: compute gamma/delta priors
    step3_params = [
        {
            "residual_variance": torch.tensor([1.0]),
            "n_samples": torch.tensor(10),
        }
        for _ in range(2)
    ]
    step3_params_local = {}
    step3_output_keys = ["sigma_hat_g"]
    step3_node_output_keys = []

    # Step 5: EB estimates
    step4_params = [
        {
            "gamma_hat_ig": torch.tensor([[0.5], [0.5]]),
            "delta_hat_ig": torch.tensor([[0.5], [0.5]]),
        }
        for _ in range(2)
    ]
    step4_params_local = {
        "standardize_result": True,
    }
    step4_output_keys = [
        "gamma_bar",
        "tau_2",
        "lambda_bar_i",
        "theta_bar_i",
        "standardize_result",
    ]
    step4_node_output_keys = []

    step_params_list = [
        step1_params,
        step2_params,
        step3_params,
        step4_params,
    ]
    step_params_local_list = [
        step1_params_local,
        step2_params_local,
        step3_params_local,
        step4_params_local,
    ]
    step_output_keys = [
        step1_output_keys,
        step2_output_keys,
        step3_output_keys,
        step4_output_keys,
    ]
    step_node_output_keys = [
        step1_node_output_keys,
        step2_node_output_keys,
        step3_node_output_keys,
        step4_node_output_keys,
    ]

    monkeypatch = MonkeyPatch()
    # Patch FederatedAnalytics used inside _FedCombatParameters so that
    # internal calls use the mocked analytics instance.items
    monkeypatch.setattr(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat_parameters.FederatedAnalytics",  # noqa: E501
        lambda *args, **kwargs: mock_analytics,
    )
    # Patch _FedCombatTrainModel used inside _FedCombatParameters so that
    # internal calls use the mocked training model instance.
    monkeypatch.setattr(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat_parameters._FedCombatTrainModel",  # noqa: E501
        lambda *args, **kwargs: mock_fedcombat_tm,
    )

    for step_index in range(1, 5):
        step_params = step_params_list[step_index - 1]
        step_params_local = step_params_local_list[step_index - 1]
        step_output, step_node_output = fedcombat_params(
            HarmonizationStep(step_index),
            step_params,
            step_params_local,
        )

        assert isinstance(step_output, dict)
        assert set(step_output.keys()) == set(step_output_keys[step_index - 1])
        assert isinstance(step_node_output, dict)
        for v in step_node_output.values():
            assert isinstance(v, dict)
            assert set(v.keys()) == set(step_node_output_keys[step_index - 1])


def test_stack_dict_params_empty(fedcombat_params):
    """_stack_dict_params should handle empty input list_dict_params."""

    # list_dict_params is empty; no keys to stack
    list_dict_params = []

    stacked = fedcombat_params._stack_dict_params(list_dict_params)  # type: ignore[attr-defined]

    # Expect an empty dict when there is nothing to stack
    assert isinstance(stacked, dict)
    assert stacked == {}


def test_population_std_from_sample_variance_full_branch_coverage():
    """Covers all branches of _population_std_from_sample_variance."""

    variance = {
        "n_le_zero": 9.0,
        "n_eq_one": 4.0,
        "n_gt_one": 4.0,
    }
    count = {
        "n_le_zero": 0,
        "n_eq_one": 1,
        "n_gt_one": 5,
    }

    result = _FedCombatParameters._population_std_from_sample_variance(
        variance=variance,
        count=count,
    )

    assert math.isnan(result["n_le_zero"])
    assert result["n_eq_one"] == 0.0
    assert math.isclose(result["n_gt_one"], math.sqrt(4.0 * 4.0 / 5.0), rel_tol=1e-12)
