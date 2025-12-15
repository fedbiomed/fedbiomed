# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.analytics
"""

from ._analytics_strategy import AnalyticsStrategy
from ._image_analytics import ImageAnalytics
from ._tabular_analytics import TabularAnalytics

__all__ = [
    "AnalyticsStrategy",
    "ImageAnalytics",
    "TabularAnalytics",
]
