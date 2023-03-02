# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
The `fedbiomed.common.models` module includes model abstraction classes
that can be used either with plain framework specific models or with declearn model

declearn repository: https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/r2.1
"""
from ._model import Model
from ._sklearn import SkLearnModel, BaseSkLearnModel
from ._torch import TorchModel

__all__ = [
    "SkLearnModel",
    "TorchModel",
    "Model",
    "BaseSkLearnModel"
]