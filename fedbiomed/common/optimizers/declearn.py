# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
"""Imports of declearn Optimizers Optimodules that are compatible with FedBioMed"""

from typing import Dict

from declearn.optimizer.modules import (
    AdaGradModule,
    AdamModule,
    AuxVar,
    EWMAModule,
    MomentumModule,
    OptiModule,
    RMSPropModule,
    ScaffoldAuxVar,
    ScaffoldClientModule,
    ScaffoldServerModule,
    YogiModule,
    YogiMomentumModule,
)
from declearn.optimizer.regularizers import (
    FedProxRegularizer,
    LassoRegularizer,
    Regularizer,
    RidgeRegularizer,
)
from declearn.utils import get_device_policy, set_device_policy

__all__ = [
    "FedProxRegularizer",
    "LassoRegularizer",
    "RidgeRegularizer",
    "AdaGradModule",
    "AdamModule",
    "AuxVar",
    "EWMAModule",
    "RMSPropModule",
    "MomentumModule",
    "ScaffoldAuxVar",
    "ScaffoldClientModule",
    "ScaffoldServerModule",
    "YogiModule",
    "YogiMomentumModule",
    "get_device_policy",
    "set_device_policy",
]

_REGULARIZERS = (
    FedProxRegularizer,
    LassoRegularizer,
    RidgeRegularizer,
)

_MODULES = (
    AdaGradModule,
    AdamModule,
    EWMAModule,
    RMSPropModule,
    MomentumModule,
    ScaffoldClientModule,
    ScaffoldServerModule,
    YogiModule,
    YogiMomentumModule,
)


def list_optim_regularizers() -> Dict[str, Regularizer]:
    """Returns list of available `declearn` `Regularizer` that are compatible with Fed-BioMed framework.

    Returns:
        Dict[str, Regularizer]: Mapping of <regularizer name, Regularizer class>
    """
    return {r.name: r for r in _REGULARIZERS}


def list_optim_modules() -> Dict[str, OptiModule]:
    """Returns a dictionary of all available OptiModules of `Declearn` compatible
    with Fed-BioMed frameworks.

    `OptiModule` is a `declearn` class for Optimizer modules in `declearn` package.

    Returns:
        Mapping of <modules names, modules class>
    """
    return {m.name: m for m in _MODULES}
