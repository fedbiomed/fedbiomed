"""Tests for _FedCombatParameters computing parameters for FedComBat."""

import pytest
import torch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.node.jobs._fedcombat_jobs import _FedCombatJobs


def test_fedcombat_jobs_compute():
    fedcombat_jobs = _FedCombatJobs()

    # Step 1: standardize data
    step1_params = {
        "global_mean_covariates": torch.tensor([1.0]),
        "global_mean_phenotypes": torch.tensor([1.0]),
        "global_std_covariates": torch.tensor([1.0]),
        "global_std_phenotypes": torch.tensor([1.0]),
    }
    step1_output_keys = {}

    # Step 2: train model - no job
    step2_params = {}
    step2_output_keys = {}

    # Step 3: compute residual variance
    step3_params = {
        "biological_model": torch.tensor([[1.0]]),
        "global_bias_model": torch.tensor([[1.0]]),
    }
    step3_output_keys = {"residual_variance", "n_samples"}

    # Step 4: compute std residuals
    step4_params = {
        "biological_model": torch.tensor([[1.0]]),
        "global_bias_model": torch.tensor([[1.0]]),
        "sigma_hat_g": torch.tensor([1.0]),
    }
    step4_output_keys = {"gamma_hat_ig", "delta_hat_ig"}

    # Step 5: compute fedcombat params
    step5_params = {
        "biological_model": torch.tensor([[1.0]]),
        "global_bias_model": torch.tensor([[1.0]]),
        "gamma_bar": torch.tensor([1.0]),
        "tau_2": torch.tensor([1.0]),
        "lambda_bar_i": torch.tensor([1.0]),
        "theta_bar_i": torch.tensor([1.0]),
        "sigma_hat_g": torch.tensor([1.0]),
    }
    step5_output_keys = {"harmonized_dataset_id"}

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

    for step_index in [1, 3, 4, 5]:
        step_params = step_params_list[step_index - 1]
        step_output = fedcombat_jobs(HarmonizationStep(step_index), "csv", step_params)

        assert isinstance(step_output, dict)
        assert set(step_output.keys()) == set(step_output_keys[step_index - 1])


def test_fedcombat_jobs_bad_dataset_type():
    """Test that _FedCombatJobs raises an error when called with an unsupported dataset type."""
    fedcombat_jobs = _FedCombatJobs()

    # Use valid step and parameters but unsupported dataset type
    with pytest.raises(ValueError):
        fedcombat_jobs(
            HarmonizationStep(1),
            "parquet",
            {},
        )
