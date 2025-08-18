# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes for dataset's data types and structures
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from fedbiomed.common.constants import _BaseEnum


# === Enums ===
class DataReturnFormat(_BaseEnum):
    # Nota: `value` serves in `isinstance` call for validation of `transforms`
    SKLEARN = (np.ndarray, pd.DataFrame, pd.Series)
    TORCH = torch.Tensor


# === Type alias ===
Transform = Optional[Union[Callable, Dict[str, Callable]]]
