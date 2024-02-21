# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer is an interface that enables the use of [declearn](https://gitlab.inria.fr/magnet/declearn/declearn2) 's
optimizers for Federated Learning inside Fed-BioMed
"""

from .optimizer import Optimizer
from .generic_optimizers import (SklearnOptimizerProcessing,
                                 NativeSkLearnOptimizer,
                                 NativeTorchOptimizer,
                                 BaseOptimizer, DeclearnOptimizer)
from ._secagg import (
    EncryptedAuxVar,
    flatten_auxvar_for_secagg,
    unflatten_auxvar_after_secagg,
)

__all__ = [
    "Optimizer",
    "SklearnOptimizerProcessing",
    "NativeSkLearnOptimizer",
    "NativeTorchOptimizer",
    "BaseOptimizer",
    "DeclearnOptimizer",
    "EncryptedAuxVar",
    "flatten_auxvar_for_secagg",
    "unflatten_auxvar_after_secagg",
]
