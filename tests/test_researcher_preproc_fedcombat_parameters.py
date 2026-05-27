"""Tests for _FedCombatParameters computing parameters for FedComBat."""

from unittest.mock import MagicMock

import pytest
import torch
from pytest import MonkeyPatch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.researcher.datasets import FederatedDataset
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
    def fetch_stats_side_effect(stat_names):
        result = {
            "mean": {
                "n1": torch.tensor([1.0]),
                "n2": torch.tensor([2.0]),
            },
            "variance": {
                "n1": torch.tensor([1.0]),
                "n2": torch.tensor([4.0]),
            },
            "std": {
                "n1": torch.tensor([1.0]),
                "n2": torch.tensor([4.0]),
            },
        }
        return MockStatsResult(result)

    analytics.fetch_stats.side_effect = fetch_stats_side_effect
    return analytics


@pytest.fixture
def mock_fedcombat_tm():
    return MagicMock(spec=_FedCombatTrainModel)


@pytest.fixture
def fedcombat_params(mock_fds, mock_reqs, tmp_path):
    return _FedCombatParameters(
        experiment_id="exp1",
        researcher_id="r1",
        fds=mock_fds,
        reqs=mock_reqs,
        experimentation_folder=str(tmp_path),
        nodes=["n1", "n2"],
    )


def test_fedcombat_parameters_compute(
    mock_fedcombat_tm, mock_analytics, fedcombat_params
):
    # Step 1: compute mean and variance
    step1_params = [
        {
            "covariates": ["cov1", "cov2"],
            "phenotypes": ["phen1"],
        }
        for _ in range(2)
    ]
    step1_output_keys = [
        "global_mean_covariates",
        "global_mean_phenotypes",
        "global_std_covariates",
        "global_std_phenotypes",
    ]

    # Step 2: train model
    step2_params = [
        {
            "covariates": ["cov1", "cov2"],
            "phenotypes": ["phen1"],
            "training_args": {"epochs": 5},
            "model_args": {"hidden_size": 10},
            "rounds": 3,
        }
        for _ in range(2)
    ]
    step2_output_keys = [
        "biological_model",
        "global_bias_model",
    ]

    # Step 3: fit L/S model
    step3_params = [{} for _ in range(2)]
    step3_output_keys = []

    # Step 4: compute gamma/delta priors
    step4_params = [
        {
            "residual_variance": torch.tensor([1.0]),
            "n_samples": torch.tensor(10),
        }
        for _ in range(2)
    ]
    step4_output_keys = ["sigma_hat_g"]

    # Step 5: EB estimates
    step5_params = [
        {
            "gamma_hat_ig": torch.tensor([[0.5], [0.5]]),
            "delta_hat_ig": torch.tensor([[0.5], [0.5]]),
        }
        for _ in range(2)
    ]
    step5_output_keys = ["gamma_bar", "tau_2", "lambda_bar_i", "theta_bar_i"]

    step_params_list = [
        step1_params,
        step2_params,
        step3_params,
        step4_params,
        step5_params,
    ]
    step_output_keys = [
        step1_output_keys,
        step2_output_keys,
        step3_output_keys,
        step4_output_keys,
        step5_output_keys,
    ]

    monkeypatch = MonkeyPatch()
    # Patch FederatedAnalytics used inside _FedCombatParameters so that
    # internal calls use the mocked analytics instance.
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

    for step_index in range(1, 6):
        step_params = step_params_list[step_index - 1]
        step_output = fedcombat_params(HarmonizationStep(step_index), step_params)

        assert isinstance(step_output, dict)
        assert set(step_output.keys()) == set(step_output_keys[step_index - 1])


def test_stack_dict_params_empty(fedcombat_params):
    """_stack_dict_params should handle empty input list_dict_params."""

    # list_dict_params is empty; no keys to stack
    list_dict_params = []

    stacked = fedcombat_params._stack_dict_params(list_dict_params)  # type: ignore[attr-defined]

    # Expect an empty dict when there is nothing to stack
    assert isinstance(stacked, dict)
    assert stacked == {}
