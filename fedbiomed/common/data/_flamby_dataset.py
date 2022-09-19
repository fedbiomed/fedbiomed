from importlib import import_module
from enum import Enum
from typing import List

from torch.utils.data import Dataset

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedLoadingBlockError
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes
from fedbiomed.common.data._data_loading_plan import DataLoadingPlanMixin, DataLoadingBlock


class FlambyLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    FLAMBY_DATASET: str = 'flamby_dataset'
    FLAMBY_CENTER_ID: str = 'flamby_center_id'


class FlambyDatasetSelectorLoadingBlock(DataLoadingBlock):
    def __init__(self):
        super(FlambyDatasetSelectorLoadingBlock, self).__init__()
        self.flamby_dataset_name = None

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingBlock.
        """
        ret = super(FlambyDatasetSelectorLoadingBlock, self).serialize()
        ret.update({'flamby_dataset_name': self.flamby_dataset_name})
        return ret

    def deserialize(self, load_from: dict) -> DataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super(FlambyDatasetSelectorLoadingBlock, self).deserialize(load_from)
        self.flamby_dataset_name = load_from['flamby_dataset_name']
        return self

    def apply(self) -> str:
        if self.flamby_dataset_name is None:
            msg = f"{ErrorNumbers.FB316}. Attempting to read Flamby dataset name, but it was never set to any value."
            logger.critical(msg)
            raise FedbiomedLoadingBlockError(msg)
        return self.flamby_dataset_name


class FlambyCenterIDLoadingBlock(DataLoadingBlock):
    def __init__(self):
        super(FlambyCenterIDLoadingBlock, self).__init__()
        self.flamby_center_id = None

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingBlock.
        """
        ret = super(FlambyCenterIDLoadingBlock, self).serialize()
        ret.update({'flamby_center_id': self.flamby_center_id})
        return ret

    def deserialize(self, load_from: dict) -> DataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super(FlambyCenterIDLoadingBlock, self).deserialize(load_from)
        self.flamby_center_id = load_from['flamby_center_id']
        return self

    def apply(self) -> int:
        if self.flamby_center_id is None:
            msg = f"{ErrorNumbers.FB316}. Attempting to read Flamby center id, but it was never set to any value."
            logger.critical(msg)
            raise FedbiomedLoadingBlockError(msg)
        return self.flamby_center_id


class FlambyDataset(DataLoadingPlanMixin, Dataset):
    def __init__(self):
        super().__init__()
        self.__flamby_fed_class = None
        self._transform = None

    def _init_flamby_fed_class(self):
        if self.__flamby_fed_class is not None:
            msg = f"{ErrorNumbers.FB616.value}. init_flamby_fed_class may only be called once."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        if self._dlp is None or FlambyLoadingBlockTypes.FLAMBY_DATASET not in self._dlp:
            msg = f"{ErrorNumbers.FB315.value}. Flamby datasets must have an associated DataLoadingPlan containing a " \
                  f"FLAMBY_DATASET loading block. Something went wrong while saving/loading the dataset."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        flamby_module_name = f"flamby.datasets.{self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET)}"
        try:
            module = import_module(f".{self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET)}",
                                   package='flamby.datasets')
        except ModuleNotFoundError as e:
            msg = f"{ErrorNumbers.FB317.value}: Error while importing FLamby dataset package; {str(e)}"
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        center_id = self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_CENTER_ID)

        if self._transform is not None:
            self.__flamby_fed_class = module.FedClass(transform=self._transform, center=center_id, train=True,
                                                      pooled=False)
        else:
            self.__flamby_fed_class = module.FedClass(center=center_id, train=True, pooled=False)

    def get_flamby_fed_class(self):
        return self.__flamby_fed_class

    def __getitem__(self, item):
        return self.__flamby_fed_class[item]

    def __len__(self):
        return len(self.__flamby_fed_class)

    def shape(self) -> List[int]:
        return [len(self)] + list(self.__getitem__(0)[0].shape)

    def set_dlp(self, dlp):
        super().set_dlp(dlp)
        self._init_flamby_fed_class()

    def set_transform(self, transform):
        self._transform = transform
