# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
"""Imports of declearn Optimizers Optimodules that are compatible with FedBioMed
"""

from typing import Dict
from declearn.optimizer.modules import OptiModule
from declearn.optimizer.modules import (
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


__all__ = [
    "AdaGradModule",
    "AdamModule",
    "EWMAModule",
    "RMSPropModule",
    "MomentumModule",
    "ScaffoldClientModule",
    "ScaffoldServerModule",
    "YogiModule",
    "YogiMomentumModule"
]


_MODULES = (AdaGradModule,
            AdamModule,
            EWMAModule,
            RMSPropModule,
            MomentumModule,
            ScaffoldClientModule,
            ScaffoldServerModule,
            YogiModule,
            YogiMomentumModule,
            )


def list_optim_modules() -> Dict[str, OptiModule]:
    """Returns a dictionary of all available OptiModules of `Declearn` compatible 
    with Fed-BioMed frameworks.

    `OptiModule` is a `declearn` class for Optimizer modules in `declearn` package.

    Returns:
        Mapping of <modules names, modules class>
    """
    return {m.name: m for m in _MODULES}
