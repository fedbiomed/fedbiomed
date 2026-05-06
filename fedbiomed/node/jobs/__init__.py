# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.node.jobs package
"""

from ._fa_job import FAJob
from ._preproc_job import PreprocJob

__all__ = [
    "FAJob",
    "PreprocJob",
]
