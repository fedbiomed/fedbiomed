"""
The `fedbiomed.common.metrics` module includes performance metrics class used for training/testing evaluation.
This module applies sklearn.metrics after transforming input array into acceptable input types.
"""

from .metrics import Metrics

__all__ = [
    "Metrics"
]