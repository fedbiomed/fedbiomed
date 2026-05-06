import pytest
import torch

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.preproc import FedCombatModelWrapper


@pytest.fixture
def fedcombat_wrapper():
    n_covariates = 5
    n_phenotypes = 3
    biological_model = torch.nn.Linear(n_covariates, n_phenotypes, bias=False)
    return FedCombatModelWrapper(biological_model, n_covariates, n_phenotypes)


def test_fedcombat_wrapper_instantiate_ok(fedcombat_wrapper):
    fedcombat_wrapper.forward(torch.randn(10, 5))

    assert isinstance(fedcombat_wrapper, FedCombatModelWrapper)


def test_fedcombat_wrapper_raise_on_bias():
    n_covariates = 5
    n_phenotypes = 3
    biological_model = torch.nn.Linear(n_covariates, n_phenotypes, bias=True)

    with pytest.raises(FedbiomedExperimentError):
        FedCombatModelWrapper(biological_model, n_covariates, n_phenotypes)


def test_fedcombat_wrapper_raise_on_non_zero_bias():
    n_covariates = 5
    n_phenotypes = 3
    biological_model = torch.nn.Linear(n_covariates, n_phenotypes, bias=False)

    # Manually add a bias parameter without using bias=True
    biological_model.contain_bias_in_name = torch.nn.Parameter(
        torch.zeros(n_phenotypes)
    )

    with pytest.raises(FedbiomedExperimentError):
        FedCombatModelWrapper(biological_model, n_covariates, n_phenotypes)
