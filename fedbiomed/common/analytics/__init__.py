# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.analytics
"""

from .analytics_strategy import AnalyticsStrategy
from .image_analytics import ImageAnalytics
from .tabular_analytics import TabularAnalytics

__all__ = [
    "AnalyticsStrategy",
    "ImageAnalytics",
    "TabularAnalytics",
]
