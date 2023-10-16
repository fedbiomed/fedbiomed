# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
"""Imports of declearn Optimizers Optimodules that are compatible with FedBioMed
"""

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