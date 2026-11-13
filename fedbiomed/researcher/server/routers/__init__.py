# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Routes of the researcher user API"""

from ._datasets import router as datasets_router
from ._health import router as health_router
from ._nodes import router as nodes_router

__all__ = ["datasets_router", "health_router", "nodes_router"]
