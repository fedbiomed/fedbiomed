# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
"""Imports of declearn regularizers that are compatible with FedBioMed
"""
from typing import Dict
from declearn.optimizer.regularizers import Regularizer

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

_REGULARIZERS = (
    FedProxRegularizer,
    LassoRegularizer,
    RidgeRegularizer,
)


def list_optim_regularizers() -> Dict[str, Regularizer]:
    """Returns list of available `declearn` `Regularizer` that are compatible with Fed-BioMed framework.

    Returns:
        Dict[str, Regularizer]: Mapping of <regularizer name, Regularizer class>
    """
    return {r.name : r for r in _REGULARIZERS}
