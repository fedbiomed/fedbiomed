import pytest
import torch

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.preproc import (
    FedCombatBiasModel,
    FedCombatBiologicalModel,
    FedCombatModelWrapper,
)

n_covariates = 5
n_phenotypes = 3


@pytest.fixture
def bias_model():
    return FedCombatBiasModel(n_phenotypes)


@pytest.fixture
def biological_model():
    return FedCombatBiologicalModel(n_covariates, n_phenotypes)


@pytest.fixture
def fedcombat_wrapper(bias_model, biological_model):
    return FedCombatModelWrapper(biological_model, bias_model)


def test_fedcombat_wrapper_instantiate_ok(fedcombat_wrapper):
    fedcombat_wrapper.forward(torch.randn(10, 5))

    assert isinstance(fedcombat_wrapper, FedCombatModelWrapper)


def test_fedcombat_wrapper_raise_on_bias(bias_model):
    biological_model = torch.nn.Linear(n_covariates, n_phenotypes, bias=True)

    with pytest.raises(FedbiomedExperimentError):
        FedCombatModelWrapper(biological_model, bias_model)


def test_fedcombat_wrapper_raise_on_non_zero_bias(bias_model):
    biological_model = torch.nn.Linear(n_covariates, n_phenotypes, bias=False)

    # Manually add a bias parameter without using bias=True
    biological_model.contain_bias_in_name = torch.nn.Parameter(
        torch.zeros(n_phenotypes)
    )

    with pytest.raises(FedbiomedExperimentError):
        FedCombatModelWrapper(biological_model, bias_model)
