from .analytics_strategy import AnalyticsStrategy


class ImageAnalytics(AnalyticsStrategy):
    def mean(self, dataset, params=None):
        raise NotImplementedError
