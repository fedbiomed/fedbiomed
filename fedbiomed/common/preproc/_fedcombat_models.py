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
