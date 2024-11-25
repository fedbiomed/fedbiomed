# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
"""Imports of declearn Optimizers Optimodules that are compatible with FedBioMed
"""

from typing import Dict

from declearn.utils import set_device_policy, get_device_policy
from declearn.optimizer.regularizers import (
    Regularizer,
    FedProxRegularizer,
    LassoRegularizer,
    RidgeRegularizer,
)

from declearn.optimizer.modules import (
    AuxVar,
    OptiModule,
    AdaGradModule,
    AdamModule,
    EWMAModule,
    RMSPropModule,
    MomentumModule,
    ScaffoldAuxVar,
    ScaffoldClientModule,
    ScaffoldServerModule,
    YogiModule,
    YogiMomentumModule
)


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
    "set_device_policy"
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
    YogiMomentumModule
)


def list_optim_regularizers() -> Dict[str, Regularizer]:
    """Returns list of available `declearn` `Regularizer` that are compatible with Fed-BioMed framework.

    Returns:
        Dict[str, Regularizer]: Mapping of <regularizer name, Regularizer class>
    """
    return {r.name : r for r in _REGULARIZERS}



def list_optim_modules() -> Dict[str, OptiModule]:
    """Returns a dictionary of all available OptiModules of `Declearn` compatible
    with Fed-BioMed frameworks.

    `OptiModule` is a `declearn` class for Optimizer modules in `declearn` package.

    Returns:
        Mapping of <modules names, modules class>
    """
    return {m.name: m for m in _MODULES}
