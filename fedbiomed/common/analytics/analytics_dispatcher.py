from fedbiomed.common.dataset import ImageFolderDataset, TabularDataset

from .image_analytics import ImageAnalytics
from .tabular_analytics import TabularAnalytics


class AnalyticsDispatcher:
    registry = {ImageFolderDataset: ImageAnalytics, TabularDataset: TabularAnalytics}

    @classmethod
    def register(self, dataset_cls, analytics_cls):
        self.registry[dataset_cls] = analytics_cls

    @classmethod
    def mean(self, dataset, params=None):
        analytics_cls = self.registry[type(dataset)]
        return analytics_cls().mean(dataset, params)
