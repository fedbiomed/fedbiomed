from importlib import import_module
from enum import Enum
from typing import List

from torch.utils.data import Dataset

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedLoadingBlockError
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes, DatasetTypes
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
    """A federated Flamby dataset.

    A FlambyDataset is always created in an empty state, and it **requires** a DataLoadingPlan to be finalized to a
    correct state. The DataLoadingPlan must contain at least the two following DataLoadinBlocks key-value pairs:
    - FlambyLoadingBlockTypes.FLAMBY_DATASET : FlambyDatasetSelectorLoadingBlock
    - FlambyLoadingBlockTypes.FLAMBY_CENTER_ID : FlambyCenterIDLoadingBlock

    Attributes:
        _transform: a transform function of type MonaiTransform or TorchTransform that will be applied to every sample
            when data is loaded.
        __flamby_fed_class: a private instance of the object representing the Flamby dataset, of type FedClass
    """
    def __init__(self):
        super().__init__()
        self.__flamby_fed_class = None
        self._transform = None

    def _init_flamby_fed_class(self):
        # prevent calling init on an already-initialized dataset
        if self.__flamby_fed_class is not None:
            msg = f"{ErrorNumbers.FB616.value}. Calling _init_flamby_fed_class is not allowed if the " \
                  f"__flamby_fed_class attribute is not None."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        # check that the data loading plan exists and is well-formed
        if self._dlp is None or \
                FlambyLoadingBlockTypes.FLAMBY_DATASET not in self._dlp or\
                FlambyLoadingBlockTypes.FLAMBY_CENTER_ID not in self._dlp:
            msg = f"{ErrorNumbers.FB315.value}. Flamby datasets must have an associated DataLoadingPlan containing " \
                  f"both {FlambyLoadingBlockTypes.FLAMBY_DATASET.value} and " \
                  f"{FlambyLoadingBlockTypes.FLAMBY_CENTER_ID} loading blocks. Something went wrong while " \
                  f"saving/loading the {self._dlp} associated with the dataset."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        # import the Flamby module corresponding to the dataset type
        try:
            module = import_module(f".{self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET)}",
                                   package='flamby.datasets')
        except ModuleNotFoundError as e:
            msg = f"{ErrorNumbers.FB317.value}: Error while importing FLamby dataset package; {str(e)}"
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        # set the center id
        center_id = self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_CENTER_ID)

        # finally instantiate FedClass
        if self._transform is not None:
            # Since the __init__ signatures are different, we are forced to distinguish two cases
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

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        return DatasetTypes.FLAMBY
