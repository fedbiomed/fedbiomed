import pkgutil
import types
from enum import Enum
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, Optional, Union

import torch

from fedbiomed.common.constants import DataLoadingBlockTypes, ErrorNumbers
from fedbiomed.common.dataloadingplan._data_loading_plan import (
    DataLoadingBlock,
    DataLoadingPlan,
)
from fedbiomed.common.exceptions import FedbiomedDatasetValueError, FedbiomedError
from fedbiomed.common.logger import logger

from ._controller import Controller

if find_spec("flamby") is not None:
    import flamby.datasets as flamby_datasets_module
else:
    m = (
        f"{ErrorNumbers.FB617.value}. Flamby module missing. "
        f"Install it manually with `pip install git+https://github.com/owkin/FLamby@main` or upgrade using `pip install -U flamby`."
    )
    raise ModuleNotFoundError(m)


class FlambyLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    """Additional DataLoadingBlockTypes specific to Flamby data"""

    FLAMBY_DATASET_METADATA: str = "flamby_dataset_metadata"


class FlambyDatasetMetadataBlock(DataLoadingBlock):
    """Metadata about a Flamby Dataset.

    Includes information on:
    - identity of the type of flamby dataset (e.g. fed_ixi, fed_heart, etc...)
    - the ID of the center of the flamby dataset
    """

    def __init__(self):
        super().__init__()
        self.metadata = {"flamby_dataset_name": None, "flamby_center_id": None}
        self._serialization_validator.update_validation_scheme(
            FlambyDatasetMetadataBlock._extra_validation_scheme()
        )

    def serialize(self) -> dict:
        """Serializes the class in a format similar to json.

        Returns:
             a dictionary of key-value pairs sufficient for reconstructing
             the DataLoadingBlock.
        """
        ret = super().serialize()
        ret.update(
            {
                "flamby_dataset_name": self.metadata["flamby_dataset_name"],
                "flamby_center_id": self.metadata["flamby_center_id"],
            }
        )
        return ret

    def deserialize(self, load_from: dict) -> DataLoadingBlock:
        """Reconstruct the DataLoadingBlock from a serialized version.

        Args:
            load_from: a dictionary as obtained by the serialize function.
        Returns:
            the self instance
        """
        super().deserialize(load_from)
        self.metadata["flamby_dataset_name"] = load_from["flamby_dataset_name"]
        self.metadata["flamby_center_id"] = load_from["flamby_center_id"]
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
            msg = (
                f"{ErrorNumbers.FB316}. Attempting to read Flamby dataset metadata, but "
                f"the {[k for k, v in self.metadata.items() if v is None]} keys were not previously set."
            )
            logger.critical(msg)
            raise FedbiomedError(msg)
        return self.metadata

    @classmethod
    def _validate_flamby_dataset_name(cls, name: str):
        if name not in FlambyController.discover_flamby_datasets().values():
            return (
                False,
                f"Flamby dataset name should be one of {FlambyController.discover_flamby_datasets().values()}, "
                f"instead got {name}",
            )
        return True

    @classmethod
    def _extra_validation_scheme(cls) -> dict:
        return {
            "flamby_dataset_name": {
                "rules": [
                    str,
                    FlambyDatasetMetadataBlock._validate_flamby_dataset_name,
                ],
                "required": True,
            },
            "flamby_center_id": {"rules": [int], "required": True},
        }

    def _requires_dlp(method):
        """Decorator that raises FedbiomedDatasetError if the Data Loading Plan was not set."""

        def wrapper(self, *args, **kwargs):
            if (
                self._dlp is None
                or FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA not in self._dlp
            ):
                msg = (
                    f"{ErrorNumbers.FB315.value}. Flamby datasets must have an associated DataLoadingPlan "
                    f"containing the {FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA} loading block. "
                    f"Something went wrong while saving/loading the {self._dlp} associated with the dataset."
                )
                logger.critical(msg)
                raise FedbiomedError(msg)
            return method(self, *args, **kwargs)

        return wrapper


class FlambyController(Controller):
    def __init__(
        self,
        root=None,
        module: types.ModuleType | str = None,
        center_id: int = 0,
        dlp: Optional[Union[DataLoadingPlan, str]] = None,
        validate: bool = True,
    ):
        self._controller_kwargs = None
        super().__init__()

        if isinstance(module, str):
            self._root = root

            self._module = import_module(f".{module}", package="flamby.datasets")
            self._validate_flamby_dataset_name(self._module)
            self._module = self._module.FedClass(center=0, train=True, pooled=False)
        # if type(module) is types.ModuleType or isinstance(module, type(flamby.datasets)):
        else:
            self._root = module.data_dir
            self._validate_flamby_dataset_name(module.__class__)
            self._module = module
        if dlp is not None:
            pass
            # self.set_dlp(dlp)
        self._validated: bool = False
        self._center_id = center_id

        if validate is True:
            self.validate()

    @staticmethod
    def discover_flamby_datasets() -> Dict[int, str]:
        """Automatically discover the available Flamby datasets based on the contents of the flamby.datasets module.

        Returns:
            a dictionary {index: dataset_name} where index is an int and dataset_name is the name of a flamby module
            corresponding to a dataset, represented as str. To import said module one must prepend with the correct
            path: `import flamby.datasets.dataset_name`.

        """
        dataset_list = [
            name
            for _, name, ispkg in pkgutil.iter_modules(flamby_datasets_module.__path__)
            if ispkg
        ]
        return {i: name for i, name in enumerate(dataset_list)}

    def _check_fed_class_initialization_status(
        require_initialized, require_uninitialized, message=None
    ):
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
            msg = (
                f"{ErrorNumbers.FB617.value}. Inconsistent arguments for _check_fed_class_initialization_status "
                f"decorator. Arguments require_initialized and require_uninitialized cannot both be true."
            )
            logger.critical(msg)
            raise FedbiomedDatasetValueError(msg)

    def set_dlp(self, dlp):
        """Sets the Data Loading Plan and ensures that the flamby_fed_class is initialized

        Overrides the set_dlp function from the DataLoadingPlanMixin to make sure that self._init_flamby_fed_class
        is also called immediately after.
        """
        super().set_dlp(dlp)
        # try:
        #     self._init_flamby_fed_class()
        # except FedbiomedError as e:
        #     # clean up
        #     super().clear_dlp()
        #     raise FedbiomedError from e

    # def clear_dlp(self) -> True:
    #     super().clear_dlp()
    #     self._clear()

    def _get_nontransformed_item(self, index: int) -> Dict[str, torch.Tensor]:
        # if self._validated is False:
        #     self.validate()
        return {'data': self._module[index][0], 'target': self._module[index][1]}

    # @classmethod
    def _validate_flamby_dataset_name(cls, name: str):
        if name not in FlambyController.discover_flamby_datasets().values():
            return (
                False,
                f"Flamby dataset name should be one of {FlambyController.discover_flamby_datasets().values()}, "
                f"instead got {name}",
            )
        return True

    def validate(self):
        _ = self._get_nontransformed_item(0)
        self._validated = True

        self._controller_kwargs = {
            "root": self._root,
            #    "module_name": self._module.__class__,
            "center_id": self._center_id,
            "flamby_dataset_name": str(self._module.__module__)[
                len("flamby.datasets") + 1 : -len("dataset") - 1
            ],
            "dlp": str(None),  # add afterwards dlp
        }

    def __len__(self):
        return len(self._module)
