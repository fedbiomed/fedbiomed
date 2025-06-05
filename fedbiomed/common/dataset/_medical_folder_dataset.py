# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for BIDS-like MedicalFolderDataset
"""

from ._dataset import StructuredDataset


# WIP !!!!

class MedicalFolderDataset(StructuredDataset, MedicalFolderController):
    pass
