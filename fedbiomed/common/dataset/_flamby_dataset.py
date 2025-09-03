# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0sv
import copy
from importlib import import_module
from typing import Any, Callable, Dict, Optional

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataset_controller import FlambyController
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_method_spec

from ._simple_dataset import SimpleDataset


class FlambyDataset(SimpleDataset):
    _controller_cls = FlambyController

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        metadata: Optional[Dict] = None,
    ):
        # target_transform unused
        super().__init__()
        self.__flamby_fed_class = None
        self._transform = transform

        if metadata is None:
            # load metadata from DLP
            pass

        # import the Flamby module corresponding to the dataset type
        # metadata = self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA)

        # try:
        #     module = import_module(
        #         f".{metadata['flamby_dataset_name']}", package="flamby.datasets"
        #     )
        # except ModuleNotFoundError as e:
        #     msg = f"{ErrorNumbers.FB317.value}: Error while importing FLamby dataset package; {str(e)}"
        #     logger.critical(msg)
        #     raise FedbiomedError(msg) from e

        # # set the center id
        # center_id = metadata["flamby_center_id"]

        # # finally instantiate FedClass
        # try:
        #     if "transform" in get_method_spec(module.FedClass):
        #         # Since the __init__ signatures are different, we are forced to distinguish two cases
        #         self.__flamby_fed_class = module.FedClass(
        #             transform=self._transform,
        #             center=center_id,
        #             train=True,
        #             pooled=False,
        #         )
        #     else:
        #         self.__flamby_fed_class = module.FedClass(
        #             center=center_id, train=True, pooled=False
        #         )
        # except Exception as e:
        #     msg = f"{ErrorNumbers.FB617.value}. Error while instantiating FedClass from module {module} because of {e}"
        #     logger.critical(msg)
        #     raise FedbiomedError(msg) from e

    def _init_controller(self, controller_kwargs: Dict[str, Any]):
        # import the Flamby module corresponding to the dataset type
        # metadata = self.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA)

        try:
            module = import_module(
                f".{controller_kwargs['flamby_dataset_name']}",
                package="flamby.datasets",
            )
            # self.__flamby_fed_class = module
        except ModuleNotFoundError as e:
            msg = f"{ErrorNumbers.FB317.value}: Error while importing FLamby dataset package; {str(e)}"
            logger.critical(msg)
            raise FedbiomedError(msg) from e

        # set the center id
        center_id = controller_kwargs["center_id"]

        # finally instantiate FedClass
        try:
            if "transform" in get_method_spec(module.FedClass):
                # Since the __init__ signatures are different, we are forced to distinguish two cases
                module = module.FedClass(
                    transform=self._transform,
                    center=center_id,
                    train=True,
                    pooled=False,
                )
            else:
                module = module.FedClass(center=center_id, train=True, pooled=False)

        except Exception as e:
            msg = f"{ErrorNumbers.FB617.value}. Error while instantiating FedClass from module {module} because of {e}"
            logger.critical(msg)
            raise FedbiomedError(msg) from e

        controller_kwargs = copy.deepcopy(controller_kwargs)
        controller_kwargs["module"] = module
        controller_kwargs.pop("flamby_dataset_name")
        super()._init_controller(controller_kwargs=controller_kwargs)

    def complete_initialization(
        self,
        controller_kwargs: Dict[str, Any],
        to_format: DataReturnFormat = None,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated to expected return format
        """
        if to_format is not None and to_format != DataReturnFormat.TORCH:
            raise FedbiomedValueError(f"Flamby only works with {DataReturnFormat.TORCH.value}, but got {to_format}")
        self._init_controller(controller_kwargs=controller_kwargs)
        # Recover sample and validate consistency of transforms
        sample = self._controller._get_nontransformed_item(0)
        # self._validate_pipeline(
        #     {modality: sample[modality] for modality in self._data_modalities},
        #     transform=self._transform,
        # )
        # self._validate_pipeline(
        #     {modality: sample[modality] for modality in self._target_modalities},
        #     transform=self._target_transform,
        #     is_target=True,
        # )

    def _validate_pipeline(self, data=None, transform=None, is_target=False):
        if transform is not None:
            self._controller._module._transform = transform

    def get_flamby_fed_class(self):
        """Returns the instance of the wrapped Flamby FedClass"""
        return self.__flamby_fed_class

    def _clear(self):
        """Clears the wrapped FedClass and the associated transforms"""
        self.__flamby_fed_class = None
        self._transform = None

    def __getitem__(self, item):
        """Forwards call to the flamby_fed_class"""
        try:
            # return self.__flamby_fed_class[item]
            return self._controller._get_nontransformed_item(item)
        except AssertionError as aerr:
            raise FedbiomedValueError(f"In FlambyDataset, {aerr}") from aerr

    def __len__(self):
        """Forwards call to the flamby_fed_class"""
        return len(self.__flamby_fed_class)

    def shape(self):
        return [len(self)] + list(self[0][0].shape)

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        """Returns the Flamby DatasetType"""
        return DatasetTypes.FLAMBY
