# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List


class EncrypterBase(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def protect(
        self,
        t,
        vector: List[int],
    ) -> List[int]:
        """Protects given vector of integers"""
        pass


    @abstractmethod
    def aggregate(
        self,
        vectors: List[List[int]]
    ) -> List[int]:
        """Agggregate method"""
        pass
