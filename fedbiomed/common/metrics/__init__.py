"""
The `fedbiomed.common.metrics` module includes performance metrics class used for training/testing evaluation.
This module applies sklearn.metrics after transforming input array into acceptable input types.
"""

from sklearn.metrics import *
from sklearn import metrics as m
__all__ = m.__all__