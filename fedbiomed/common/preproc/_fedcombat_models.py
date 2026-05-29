# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn


class FedCombatBiologicalModel(nn.Module):
    def __init__(
        self,
        n_covariates: int,
        n_phenotypes: int,
    ):
        super().__init__()
        self.linear = nn.Linear(n_covariates, n_phenotypes, bias=False)

    def forward(self, x):
        return self.linear(x)


class FedCombatBiasModel(nn.Module):
    def __init__(
        self,
        n_phenotypes: int,
    ):
        super().__init__()
        self.bias = nn.Linear(1, n_phenotypes, bias=False)

    def forward(self, x):
        bias_column = x.new_ones((x.shape[0], 1))
        return self.bias(bias_column)
