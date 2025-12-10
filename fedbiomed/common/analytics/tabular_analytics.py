from typing import Dict

from fedbiomed.common.analytics import AnalyticsStrategy


class TabularAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on tabular datasets"""

    def mean(self, **kwargs) -> Dict:
        return super().mean(**kwargs)
