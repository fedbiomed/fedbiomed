from importlib import import_module
from enum import Enum
from typing import List, Dict, Union
import pkgutil
import inspect

from torch.utils.data import Dataset
import flamby.datasets as flamby_datasets_module
from torchvision.transforms import Compose as TorchCompose
from monai.transforms import Compose as MonaiCompose


from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedLoadingBlockError, FedbiomedDatasetValueError
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes, DatasetTypes
from fedbiomed.common.data._data_loading_plan import DataLoadingPlanMixin, DataLoadingBlock


def discover_flamby_datasets() -> Dict[int, str]:
    """Automatically discover the available Flamby datasets based on the contents of the flamby.datasets module.

    Returns:
        a dictionary {index: dataset_name} where index is an int and dataset_name is the name of a flamby module
        corresponding to a dataset, represented as str. To import said module one must prepend with the correct
        path: `import flamby.datasets.dataset_name`.

    """
    dataset_list = [name for _, name, ispkg in pkgutil.iter_modules(flamby_datasets_module.__path__) if ispkg]
    return {i: name for i, name in enumerate(dataset_list)}


class FlambyLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    """Additional DataLoadingBlockTypes specific to Flamby data"""
    FLAMBY_DATASET: str = 'flamby_dataset'
    FLAMBY_CENTER_ID: str = 'flamby_center_id'


class FlambyDatasetSelectorLoadingBlock(DataLoadingBlock):
    """Identity of the type of flamby dataset (e.g. fed_ixi, fed_heart, etc...)"""
    def __init__(self):
        super(FlambyDatasetSelectorLoadingBlock, self).__init__()
        self.flamby_dataset_name = None
        self._serialization_validator.validation_scheme.update(
            FlambyDatasetSelectorLoadingBlock._extra_validation_scheme())

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
        """Returns the name of the selected flamby dataset.

        Note that this will be the same as the module name required to instantiate the FedClass. However, it will not
        contain the full module path, hence to properly import this module it must be prepended with
        `flamby.datasets`, for example `import flamby.datasets.flamby_dataset_name`
        """
        if self.flamby_dataset_name is None:
            msg = f"{ErrorNumbers.FB316}. Attempting to read Flamby dataset name, but it was never set to any value."
            logger.critical(msg)
            raise FedbiomedLoadingBlockError(msg)
        return self.flamby_dataset_name

    @classmethod
    def _validate_flamby_dataset_name(cls, name: str):
        if name not in discover_flamby_datasets().values():
            return False, f"Flamby dataset name should be one of {discover_flamby_datasets().values()}, " \
                          f"instead got {name}"
        return True

    @classmethod
    def _extra_validation_scheme(cls) -> dict:
        return {
            'flamby_dataset_name': {
                'rules': [str, FlambyDatasetSelectorLoadingBlock._validate_flamby_dataset_name],
                'required': True
            }
        }


class FlambyCenterIDLoadingBlock(DataLoadingBlock):
    """The ID of the center in a flamby dataset."""
    def __init__(self):
        super(FlambyCenterIDLoadingBlock, self).__init__()
        self.flamby_center_id = None
        self._serialization_validator.validation_scheme.update(
            FlambyCenterIDLoadingBlock._extra_validation_scheme())

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
        """Returns the ID of the center (int) as selected when the DataLoadingPlan was created."""
        if self.flamby_center_id is None:
            msg = f"{ErrorNumbers.FB316}. Attempting to read Flamby center id, but it was never set to any value."
            logger.critical(msg)
            raise FedbiomedLoadingBlockError(msg)
        return self.flamby_center_id

    @classmethod
    def _extra_validation_scheme(cls) -> dict:
        return {
            'flamby_center_id': {
                'rules': [int],
                'required': True
            }
        }


class FlambyDataset(DataLoadingPlanMixin, Dataset):
    """A federated Flamby dataset.

    A FlambyDataset is a wrapper around a flamby FedClass instance, adding functionalities and interfaces that are
    specific to Fed-BioMed.

    A FlambyDataset is always created in an empty state, and it **requires** a DataLoadingPlan to be finalized to a
    correct state. The DataLoadingPlan must contain at least the two following DataLoadinBlocks key-value pairs:
    - FlambyLoadingBlockTypes.FLAMBY_DATASET : FlambyDatasetSelectorLoadingBlock
    - FlambyLoadingBlockTypes.FLAMBY_CENTER_ID : FlambyCenterIDLoadingBlock

    The lifecycle of the DataLoadingPlan and the wrapped FedClass are tightly interlinked: when the DataLoadingPlan
    is set, the wrapped FedClass is initialized and instantiated. When the DataLoadingPlan is cleared, the wrapped
    FedClass is also cleared. Hence, an invariant of this class is that the self._dlp and self.__flamby_fed_class
    should always be either both None, or both set to some value.

    Attributes:
        _transform: a transform function of type MonaiTransform or TorchTransform that will be applied to every sample
            when data is loaded.
        __flamby_fed_class: a private instance of the wrapped Flamby FedClass
    """
    def __init__(self):
        super().__init__()
        self.__flamby_fed_class = None
        self._transform = None

    def _init_flamby_fed_class(self) -> None:
        """Initializes once the __flamby_fed_class attribute with an object of type FedClass.

        This function cannot be called multiple times. It sets the self.__flamby_fed_class attribute by extracting
        the necessary information from the DataLoadingPlan. Therefore, a DataLoadingPlan is **required** to
        correctly use the FlambyDataset class. See the
        [FlambyDataset][fedbiomed.common.data._flamby_dataset.FlambyDataset] documentation for more details.

        The correct FedClass constructor will be automatically called according to whether the transform attribute
        was set in the class.

        Raises:
            FedbiomedDatasetError: if one of the following conditions occurs
                - __flamby_fed_class is not None (i.e. the function was already called)
                - the Data Loading Plan is not present or malformed
                - the Flamby dataset module could not be loaded
        """
        # prevent calling init on an already-initialized dataset
        if self.__flamby_fed_class is not None:
            msg = f"{ErrorNumbers.FB616.value}. Calling _init_flamby_fed_class is not allowed if the " \
                  f"__flamby_fed_class attribute has already been initialized."
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
        try:
            if 'transform' in inspect.signature(module.FedClass).parameters.keys():
                # Since the __init__ signatures are different, we are forced to distinguish two cases
                self.__flamby_fed_class = module.FedClass(transform=self._transform, center=center_id, train=True,
                                                          pooled=False)
            else:
                self.__flamby_fed_class = module.FedClass(center=center_id, train=True, pooled=False)
        except Exception as e:
            msg = f"{ErrorNumbers.FB616.value}. Error while instantiating FedClass from module {module} because of {e}"
            logger.critical(msg)
            raise FedbiomedDatasetError(msg) from e

    def init_transform(self, transform: Union[MonaiCompose, TorchCompose]) -> Union[MonaiCompose, TorchCompose]:
        """Initializes the transform attribute. Must be called before initialization of the wrapped FedClass.


        Arguments:
            transform: a composed transform of type torchvision.transforms.Compose or monai.transforms.Compose

        Raises:
            FedbiomedDatasetError: if the wrapped FedClass was already initialized.
            FedbiomedDatasetValueError: if the input is not of the correct type.
        """
        if self.__flamby_fed_class is not None:
            msg = f"{ErrorNumbers.FB617.value}. Calling init_transform is not allowed if the wrapped FedClass has " \
                  f"already been initialized. At your own risk, you may call clear_dlp to reset the full FlambyDataset"
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        if not isinstance(transform, (MonaiCompose, TorchCompose)):
            msg = f"{ErrorNumbers.FB617.value}. FlambyDataset transform must be of type " \
                  f"torchvision.transforms.Compose or monai.transforms.Compose"
            logger.critical(msg)
            raise FedbiomedDatasetValueError(msg)

        self._transform = transform
        return self._transform

    def get_transform(self):
        """Gets the transform attribute"""
        return self._transform

    def get_flamby_fed_class(self):
        """Returns the instance of the wrapped Flamby FedClass"""
        return self.__flamby_fed_class

    def get_center_id(self) -> int:
        """Returns the center id. Requires that the DataLoadingPlan has already been set.

        Returns:
            the center id (int).
        Raises:
            FedbiomedDatasetError:
                - if the data loading plan is not set or is malformed.
                - if the wrapped FedClass is not initialized but the dlp exists
        """
        if self._dlp is None or FlambyLoadingBlockTypes.FLAMBY_CENTER_ID not in self._dlp:
            msg = f"{ErrorNumbers.FB616.value}. Flamby datasets must have an associated DataLoadingPlan containing " \
                  f"a {FlambyLoadingBlockTypes.FLAMBY_CENTER_ID} loading block in order to query its center id."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        if self.__flamby_fed_class is None:
            msg = f"{ErrorNumbers.FB616.value}. Flamby dataset is in an inconsistent state: a Data Loading Plan is " \
                  f"set but the wrapped FedClass was not initialized."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        return self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_CENTER_ID)

    def _clear(self):
        """Clears the wrapped FedClass and the associated transforms"""
        self.__flamby_fed_class = None
        self._transform = None

    def __getitem__(self, item):
        """Forwards call to the flamby_fed_class"""
        if self.__flamby_fed_class is None:
            msg = f"{ErrorNumbers.FB616.value}. Cannot get item because FedClass was not initialized."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)
        return self.__flamby_fed_class[item]

    def __len__(self):
        """Forwards call to the flamby_fed_class"""
        if self.__flamby_fed_class is None:
            msg = f"{ErrorNumbers.FB616.value}. Cannot compute len because FedClass was not initialized."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)
        return len(self.__flamby_fed_class)

    def shape(self) -> List[int]:
        """Returns the shape of the flamby_fed_class"""
        if self.__flamby_fed_class is None:
            msg = f"{ErrorNumbers.FB616.value}. Cannot compute shape because FedClass was not initialized."
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)
        return [len(self)] + list(self.__getitem__(0)[0].shape)

    def set_dlp(self, dlp):
        """Sets the Data Loading Plan and ensures that the flamby_fed_class is initialized

        Overrides the set_dlp function from the DataLoadingPlanMixin to make sure that self._init_flamby_fed_class
        is also called immediately after.
        """
        super().set_dlp(dlp)
        try:
            self._init_flamby_fed_class()
        except FedbiomedDatasetError as e:
            # clean up
            super().clear_dlp()
            raise FedbiomedDatasetError from e

    def clear_dlp(self):
        """Clears dlp and automatically clears the FedClass

        Tries to guarantee some semblance of integrity by also clearing the FedClass, since setting the dlp
        initializes it.
        """
        super().clear_dlp()
        self._clear()

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        """Returns the Flamby DatasetType"""
        return DatasetTypes.FLAMBY
