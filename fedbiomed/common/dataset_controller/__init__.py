# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset_controller
"""

from ._controller import Controller
from ._medical_folder_controller import NewMedicalFolderController
from ._mnist_controller import MnistController

__all__ = [
    "Controller",
    "NewMedicalFolderController",
    "MnistController",
]
