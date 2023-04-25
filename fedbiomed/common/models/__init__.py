# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
The `fedbiomed.common.models` module includes model abstraction classes
that can be used with plain framework specific models.

Please visit [Declearn](https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/r2.1) repository for
the "TorchVector" and "NumpyVector" classes used in this module.
"""

from ._model import Model
from ._sklearn import SkLearnModel, \
    BaseSkLearnModel, \
    MLPSklearnModel, \
    SGDRegressorSKLearnModel, \
    SGDClassifierSKLearnModel
from ._torch import TorchModel

__all__ = [
    "SkLearnModel",
    "TorchModel",
    "Model",
    "SGDRegressorSKLearnModel",
    "SGDClassifierSKLearnModel",
    "MLPSklearnModel",
    "BaseSkLearnModel"
]