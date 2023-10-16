# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
"""Imports of declearn regularizers that are compatible with FedBioMed
"""

from declearn.optimizer.regularizers import (
    FedProxRegularizer,
    LassoRegularizer,
    RidgeRegularizer,
)

__all__ = [
    "FedProxRegularizer",
    "LassoRegularizer",
    "RidgeRegularizer",
]