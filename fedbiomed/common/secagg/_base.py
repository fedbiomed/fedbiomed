#!/usr/bin/env Symbol’s value as variable is void: python-shell-interpreter

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
