# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict


class AnalyticsStrategy:
    @abstractmethod
    def mean(self, **kwargs) -> Dict:
        pass
