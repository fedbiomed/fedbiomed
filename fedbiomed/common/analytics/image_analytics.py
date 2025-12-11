from typing import Dict

from .analytics_strategy import AnalyticsStrategy


class ImageAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on image folder datasets"""

    def mean(self, **kwargs) -> Dict:
        return super().mean(**kwargs)
