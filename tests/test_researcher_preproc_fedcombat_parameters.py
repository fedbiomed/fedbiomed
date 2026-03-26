"""Tests for _FedCombatParameters computing parameters for FedComBat."""

import torch

from fedbiomed.common.constants import HarmonizationStep
from fedbiomed.researcher.federated_workflows.preproc._fedcombat_parameters import (
    _FedCombatParameters,
)


def test_fedcombat_parameters_compute():
    fedcombat_params = _FedCombatParameters()

    # Step 1: compute grand mean
    # Input: local means and sample counts from each node
    step1_params = [{} for _ in range(2)]
    step1_output_keys = []

    # Step 2: compute variance
    step2_params = [
        {
            "mean_covariates": torch.tensor([1.0]),
            "mean_phenotypes": torch.tensor([1.0]),
            "std_covariates": torch.tensor([1.0]),
            "std_phenotypes": torch.tensor([1.0]),
            "n_samples": torch.tensor(10),
        }
        for _ in range(2)
    ]
    step2_output_keys = [
        "global_mean_covariates",
        "global_mean_phenotypes",
        "global_std_covariates",
        "global_std_phenotypes",
    ]

    # Step 3: standardize data
    step3_params = [{} for _ in range(2)]
    step3_output_keys = []

    # Step 4: fit L/S model
    step4_params = [{} for _ in range(2)]
    step4_output_keys = []

    # Step 5: compute gamma/delta priors
    step5_params = [
        {
            "residual_variance": torch.tensor([1.0]),
            "n_samples": torch.tensor(10),
        }
        for _ in range(2)
    ]
    step5_output_keys = ["sigma_hat_g"]

    # Step 6: EB estimates
    step6_params = [
        {
            "gamma_hat_ig": torch.tensor([[0.5], [0.5]]),
            "delta_hat_ig": torch.tensor([[0.5], [0.5]]),
        }
        for _ in range(2)
    ]
    step6_output_keys = ["gamma_bar", "tau_2", "lambda_bar_i", "theta_bar_i"]

    step_params_list = [
        step1_params,
        step2_params,
        step3_params,
        step4_params,
        step5_params,
        step6_params,
    ]
    step_output_keys = [
        step1_output_keys,
        step2_output_keys,
        step3_output_keys,
        step4_output_keys,
        step5_output_keys,
        step6_output_keys,
    ]

    for step_index in range(1, 7):
        step_params = step_params_list[step_index - 1]
        step_output = fedcombat_params(HarmonizationStep(step_index), step_params)

        assert isinstance(step_output, dict)
        assert set(step_output.keys()) == set(step_output_keys[step_index - 1])


def test_stack_dict_params_empty():
    """_stack_dict_params should handle empty input list_dict_params."""
    fedcombat_params = _FedCombatParameters()

    # list_dict_params is empty; no keys to stack
    list_dict_params = []

    stacked = fedcombat_params._stack_dict_params(list_dict_params)  # type: ignore[attr-defined]

    # Expect an empty dict when there is nothing to stack
    assert isinstance(stacked, dict)
    assert stacked == {}
