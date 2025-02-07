# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

try:
    import flamby
except ModuleNotFoundError as e:
    from fedbiomed.common.constants import ErrorNumbers
    m = (f"{ErrorNumbers.FB617.value}. Flamby module missing. "
         f"Install it manually with `pip install git+https://github.com/owkin/FLamby@main`.")
    raise ModuleNotFoundError(m) from e


from importlib import import_module
from enum import Enum
from typing import List, Dict, Union
import pkgutil

from torch.utils.data import Dataset
import flamby.datasets as flamby_datasets_module
from torchvision.transforms import Compose as TorchCompose
from monai.transforms import Compose as MonaiCompose

from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedLoadingBlockError, FedbiomedDatasetValueError
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes, DatasetTypes
from fedbiomed.common.utils import get_method_spec
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
    FLAMBY_DATASET_METADATA: str = 'flamby_dataset_metadata'


class FlambyDatasetMetadataBlock(DataLoadingBlock):
    """Metadata about a Flamby Dataset.

    Includes information on:
    - identity of the type of flamby dataset (e.g. fed_ixi, fed_heart, etc...)
    - the ID of the center of the flamby dataset
    """
    def __init__(self):
        super().__init__()
        self.metadata = {
            "flamby_dataset_name": None,
            "flamby_center_id": None
        }
        self._serialization_validator.update_validation_scheme(
            FlambyDatasetMetadataBlock._extra_validation_scheme())

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingBlock.
        """
        ret = super().serialize()
        ret.update({'flamby_dataset_name': self.metadata['flamby_dataset_name'],
                    'flamby_center_id': self.metadata['flamby_center_id']
                    })
        return ret

    def deserialize(self, load_from: dict) -> DataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super().deserialize(load_from)
        self.metadata['flamby_dataset_name'] = load_from['flamby_dataset_name']
        self.metadata['flamby_center_id'] = load_from['flamby_center_id']
        return self

    def apply(self) -> dict:
        """Returns a dictionary of dataset metadata.

        The metadata dictionary contains:
        - flamby_dataset_name: (str) the name of the selected flamby dataset.
        - flamby_center_id: (int) the center id selected at dataset add time.

        Note that the flamby_dataset_name will be the same as the module name required to instantiate the FedClass.
        However, it will not contain the full module path, hence to properly import this module it must be
        prepended with `flamby.datasets`, for example `import flamby.datasets.flamby_dataset_name`

        Returns:
            this data loading block's metadata
        """
        if any([v is None for v in self.metadata.values()]):
            msg = f"{ErrorNumbers.FB316}. Attempting to read Flamby dataset metadata, but " \
                  f"the {[k for k,v in self.metadata.items() if v is None]} keys were not previously set."
            logger.critical(msg)
            raise FedbiomedLoadingBlockError(msg)
        return self.metadata

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
                'rules': [str, FlambyDatasetMetadataBlock._validate_flamby_dataset_name],
                'required': True
            },
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
    correct state. The DataLoadingPlan must contain at least the following DataLoadinBlock key-value pair:
    - FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA : FlambyDatasetMetadataBlock

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

    def _check_fed_class_initialization_status(require_initialized, require_uninitialized, message=None):
        """Decorator that raises FedbiomedDatasetError if the FedClass was not initialized.

        This decorator can be used as a shorthand for testing whether the self.__flamby_fed_class was correctly
        initialized before using a method of the FlambyDataset class. Note that the arguments require_initialized
        and require_uninitialized cannot both the same value.

        Arguments:
            require_initialized (bool): whether the wrapped method should only work if the FedClass has already
                been initialized
            require_uninitialized (bool): whether the wrapped method should only work if the FedClass has not yet
                been initialized
            message (str): the error message to display
        """
        if require_initialized == require_uninitialized:
            msg = f"{ErrorNumbers.FB617.value}. Inconsistent arguments for _check_fed_class_initialization_status " \
                  f"decorator. Arguments require_initialized and require_uninitialized cannot both be true."
            logger.critical(msg)
            raise FedbiomedDatasetValueError(msg)

        def decorator(method):
            def wrapper(self, *args, **kwargs):
                if (require_initialized and self.__flamby_fed_class is None) or \
                        (require_uninitialized and self.__flamby_fed_class is not None):
                    msg = f"{ErrorNumbers.FB617.value}. {message or 'Wrong FedClass initialization status.'}"
                    logger.critical(msg)
                    raise FedbiomedDatasetError(msg)
                return method(self, *args, **kwargs)
            return wrapper
        return decorator

    def _requires_dlp(method):
        """Decorator that raises FedbiomedDatasetError if the Data Loading Plan was not set."""
        def wrapper(self, *args, **kwargs):
            if self._dlp is None or FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA not in self._dlp:
                msg = f"{ErrorNumbers.FB315.value}. Flamby datasets must have an associated DataLoadingPlan " \
                      f"containing the {FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA} loading block. " \
                      f"Something went wrong while saving/loading the {self._dlp} associated with the dataset."
                logger.critical(msg)
                raise FedbiomedDatasetError(msg)
            return method(self, *args, **kwargs)
        return wrapper

    @_check_fed_class_initialization_status(require_initialized=False,
                                            require_uninitialized=True,
                                            message="Calling _init_flamby_fed_class is not allowed if the "
                                                    "__flamby_fed_class attribute has already been initialized.")
    @_requires_dlp
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
        # import the Flamby module corresponding to the dataset type
        metadata = self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA)
        try:
            module = import_module(f".{metadata['flamby_dataset_name']}",
                                   package='flamby.datasets')
        except ModuleNotFoundError as e:
            msg = f"{ErrorNumbers.FB317.value}: Error while importing FLamby dataset package; {str(e)}"
            logger.critical(msg)
            raise FedbiomedDatasetError(msg)

        # set the center id
        center_id = metadata['flamby_center_id']

        # finally instantiate FedClass
        try:
            if 'transform' in get_method_spec(module.FedClass):
                # Since the __init__ signatures are different, we are forced to distinguish two cases
                self.__flamby_fed_class = module.FedClass(transform=self._transform, center=center_id, train=True,
                                                          pooled=False)
            else:
                self.__flamby_fed_class = module.FedClass(center=center_id, train=True, pooled=False)
        except Exception as e:
            msg = f"{ErrorNumbers.FB617.value}. Error while instantiating FedClass from module {module} because of {e}"
            logger.critical(msg)
            raise FedbiomedDatasetError(msg) from e

    @_check_fed_class_initialization_status(require_initialized=False,
                                            require_uninitialized=True,
                                            message="Calling init_transform is not allowed if the wrapped FedClass "
                                                    "has already been initialized. At your own risk, you may call "
                                                    "clear_dlp to reset the full FlambyDataset")
    def init_transform(self, transform: Union[MonaiCompose, TorchCompose]) -> Union[MonaiCompose, TorchCompose]:
        """Initializes the transform attribute. Must be called before initialization of the wrapped FedClass.

        Arguments:
            transform: a composed transform of type torchvision.transforms.Compose or monai.transforms.Compose

        Raises:
            FedbiomedDatasetError: if the wrapped FedClass was already initialized.
            FedbiomedDatasetValueError: if the input is not of the correct type.
        """
        if not isinstance(transform, (MonaiCompose, TorchCompose)):
            msg = f"{ErrorNumbers.FB618.value}. FlambyDataset transform must be of type " \
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

    @_check_fed_class_initialization_status(require_initialized=True,
                                            require_uninitialized=False,
                                            message="Flamby dataset is in an inconsistent state: a Data Loading Plan "
                                                    "is set but the wrapped FedClass was not initialized.")
    @_requires_dlp
    def get_center_id(self) -> int:
        """Returns the center id. Requires that the DataLoadingPlan has already been set.

        Returns:
            the center id (int).
        Raises:
            FedbiomedDatasetError: in one of the two scenarios below
                - if the data loading plan is not set or is malformed.
                - if the wrapped FedClass is not initialized but the dlp exists
        """
        return self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA)['flamby_center_id']

    def _clear(self):
        """Clears the wrapped FedClass and the associated transforms"""
        self.__flamby_fed_class = None
        self._transform = None

    @_check_fed_class_initialization_status(require_initialized=True,
                                            require_uninitialized=False,
                                            message="Cannot get item because FedClass was not initialized.")
    def __getitem__(self, item):
        """Forwards call to the flamby_fed_class"""
        return self.__flamby_fed_class[item]

    @_check_fed_class_initialization_status(require_initialized=True,
                                            require_uninitialized=False,
                                            message="Cannot compute len because FedClass was not initialized.")
    def __len__(self):
        """Forwards call to the flamby_fed_class"""
        return len(self.__flamby_fed_class)

    @_check_fed_class_initialization_status(require_initialized=True,
                                            require_uninitialized=False,
                                            message="Cannot compute shape because FedClass was not initialized.")
    def shape(self) -> List[int]:
        """Returns the shape of the flamby_fed_class"""
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
