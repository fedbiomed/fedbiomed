# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.analytics
"""

from ._analytics_strategy import (
    AnalyticsStrategy,
    DatasetArgumentsFA,
    validate_dataset_arguments_for_fa,
)
from ._image_analytics import ImageAnalytics
from ._tabular_analytics import TabularAnalytics

__all__ = [
    "DatasetArgumentsFA",
    "AnalyticsStrategy",
    "ImageAnalytics",
    "TabularAnalytics",
    "validate_dataset_arguments_for_fa",
]
