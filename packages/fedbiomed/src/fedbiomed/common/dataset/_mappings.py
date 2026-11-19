# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Registry for dataset controllers
"""

import inspect
from typing import Dict, Optional, Tuple, Type

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataset_controller import (
    Controller,
    CustomController,
    ImageFolderController,
    MedicalFolderController,
    MedNistController,
    MnistController,
    TabularController,
)
from fedbiomed.common.exceptions import FedbiomedError

from ._custom_dataset import CustomDataset
from ._dataset import Dataset
from ._image_label_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)
from ._medical_folder_dataset import MedicalFolderDataset
from ._tabular_dataset import TabularDataset

DATASET_CLASSES_PER_TYPE: Dict[DatasetTypes, Type[Dataset]] = {
    DatasetTypes.CUSTOM: CustomDataset,  # type: ignore[type-abstract]
    DatasetTypes.IMAGES: ImageFolderDataset,
    DatasetTypes.MEDICAL_FOLDER: MedicalFolderDataset,
    DatasetTypes.MEDNIST: MedNistDataset,
    DatasetTypes.DEFAULT: MnistDataset,
    DatasetTypes.TABULAR: TabularDataset,
}


# Registry mapping data types to their controller and dataset classes
REGISTRY_CONTROLLERS: Dict[DatasetTypes, Tuple[Type[Controller], Type[Dataset]]] = {
    DatasetTypes.TABULAR: (
        TabularController,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.TABULAR],
    ),
    DatasetTypes.MEDICAL_FOLDER: (
        MedicalFolderController,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.MEDICAL_FOLDER],
    ),
    DatasetTypes.IMAGES: (
        ImageFolderController,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.IMAGES],
    ),
    DatasetTypes.DEFAULT: (
        MnistController,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.DEFAULT],
    ),
    DatasetTypes.MEDNIST: (
        MedNistController,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.MEDNIST],
    ),
    DatasetTypes.CUSTOM: (
        CustomController,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.CUSTOM],
    ),
}


def get_controller(
    data_type: str,
    controller_parameters: dict,
) -> Controller:
    """Get controller instance based on data_type and controller_parameters.

    Only the keyword arguments accepted by the controller's constructor are
    forwarded; unknown keys and `None` values are dropped.
    """
    # Validate that data_type is implemented.
    data_type_: Optional[DatasetTypes] = DatasetTypes.get_type_by_value(data_type)
    if not data_type_ or data_type_ not in REGISTRY_CONTROLLERS:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: "
            f"Unknown 'data_type', implemented are: {list(REGISTRY_CONTROLLERS.keys())}"
        )

    controller_class, _ = REGISTRY_CONTROLLERS[data_type_]

    accepted = inspect.signature(controller_class.__init__).parameters
    parameters = {
        k: v
        for k, v in controller_parameters.items()
        if k in accepted and v is not None
    }

    try:
        return controller_class(**parameters)
    except FedbiomedError:
        raise
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Unhandled exception occurred: {str(e)}"
        ) from e
