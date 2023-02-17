"""
### ** Vector Encoding Scheme **

This module encodes and decodes vector elements to create smaller size vectors using a packing technique
"""
from typing import List, Tuple
from math import ceil, floor, log2
import gmpy2


class VES:
    """The vector encoding class

    Attributes:
        ptsize: The bit length of the plaintext (the number of bits of an element in the output vector)
        addops: The number of supported addition operation on the encoded elements (to avoid overflow)
        valuesize: The bit length of an element of the input vector
        vectorsize: The number of element of the input vector
        _elementsize: The extended bit length of an element of the input vector
        _compratio: The compression ratio of the scheme
        _numbatches: The number of elements in the output vector
    """

    def __init__(
            self,
            ptsize: int,
            valuesize: int,
            vectorsize: int,
    ) -> None:
        """Vector encoder constructor

        Arg:
            _ptsize: The bit length of the plaintext (the number of bits of an element in the output vector)
            _addops: The number of supported addition operation on the encoded elements (to avoid overflow)
            _valuesize: The bit length of an element of the input vector
            _vectorsize: The number of element of the input vector
        """

        self._ptsize = ptsize
        self._valuesize = valuesize
        self._vectorsize = vectorsize
        self._numbatches = ceil(self._vectorsize / self._compratio)

    def _get_elements_size_and_compression_ratio(
            self,
            add_ops: int
    ) -> Tuple[int, int]:
        """

        Args:
            add_ops:
        """
        element_size = self._valuesize + ceil(log2(add_ops))
        comp_ratio = floor(self._ptsize / self._elementsize)

        return element_size, comp_ratio

    def encode(self, V: List[int], add_ops: int):
        """Encode a vector to a smaller size vector

        """
        element_size, comp_ratio = self._get_elements_size_and_compression_ratio(add_ops)

        bs = comp_ratio
        e = []
        E = []
        for v in V:
            e.append(v)
            bs -= 1
            if bs == 0:
                E.append(self._batch(e, element_size))
                e = []
                bs = comp_ratio
        if e:
            E.append(self._batch(e, element_size))
        return E

    def decode(self, E, add_ops: int):
        """Decode a vector back to original size vector"""

        element_size, _ = self._get_elements_size_and_compression_ratio(add_ops)

        V = []
        for e in E:
            for v in self._debatch(e, element_size):
                V.append(v)
        return V

    @staticmethod
    def _batch(
            V,
            element_size: int
    ):
        i = 0
        a = 0
        for v in V:
            a |= v << element_size * i
            i += 1
        return gmpy2.mpz(a)

    @staticmethod
    def _debatch(
            b: int,
            element_size: int
    ) -> List[int]:
        """
        """
        i = 1
        V = []
        bit = 0b1
        mask = 0b1
        for _ in range(element_size - 1):
            mask <<= 1
            mask |= bit

        while b != 0:
            v = mask & b
            V.append(int(v))
            b >>= element_size
        return V
