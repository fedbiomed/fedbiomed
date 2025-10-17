# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Registry for dataset controllers and their parameters
"""

from dataclasses import asdict, dataclass, fields
from typing import Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset_controller import (
    Controller,
    ImageFolderController,
    MedicalFolderController,
    MedNistController,
    MnistController,
    TabularController,
)
from fedbiomed.common.exceptions import FedbiomedError


@dataclass
class ControllerParametersBase:
    root: str

    def to_dict(self) -> dict:
        """Convert entry to dictionary - removes None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


@dataclass
class MedicalFolderParameters(ControllerParametersBase):
    tabular_file: Optional[str]
    index_col: Optional[str]
    dlp: Optional[DataLoadingPlan]


# Registry mapping data types to corresponding controller and expected parameters
REGISTRY_CONTROLLERS = {
    "csv": (TabularController, ControllerParametersBase),
    "medical-folder": (MedicalFolderController, MedicalFolderParameters),
    "images": (ImageFolderController, ControllerParametersBase),
    "default": (MnistController, ControllerParametersBase),
    "mednist": (MedNistController, ControllerParametersBase),
}


def get_controller(
    data_type: str,
    controller_parameters: dict,
) -> Controller:
    """Get controller instance based on data_type and dataset_parameters"""
    # Validate that data_type is implemented.
    if data_type not in REGISTRY_CONTROLLERS:
        raise NotImplementedError(
            f"Unknown 'data_type', implemented are: {list(REGISTRY_CONTROLLERS.keys())}"
        )

    controller_class, parameters_class = REGISTRY_CONTROLLERS[data_type]

    # Validate and instantiate parameters
    try:
        parameters_instance = parameters_class.from_dict(controller_parameters)
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Failed to parse dataset_parameters: {str(e)}"
        ) from e

    try:
        return controller_class(**parameters_instance.to_dict())
    except FedbiomedError:
        raise
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Unhandled exception occurred: {str(e)}"
        ) from e
