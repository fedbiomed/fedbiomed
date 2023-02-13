CLIPPING_RANGE = 3
TARGET_RANGE = 10000

import random

import gmpy2
import numpy as np

from typing import List, Optional


def _check_clipping_range(weight: List[float], clipping_range: float):
    for x in weight:
        if x < -clipping_range or x > clipping_range:
            print(
                f"There are some numbers in the local vector that exceeds clipping range. Please increase the "
                f"clipping range to account for value {x}",
            )
            return


def quantize(
    weight: List[float],
    clipping_range: Optional[float] = None,
    target_range: Optional[int] = None,
) -> np.ndarray:
    """
    Quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776
    Return a vector in the range [0, target_range-1]
    """
    if clipping_range is None:
        clipping_range = CLIPPING_RANGE
    if target_range is None:
        target_range = TARGET_RANGE
    _check_clipping_range(weight, clipping_range)
    f = np.vectorize(
        lambda x: min(
            target_range - 1,
            (sorted((-clipping_range, x, clipping_range))[1] + clipping_range)
            * target_range
            / (2 * clipping_range),
        )
    )
    quantized_list = f(weight).astype(int)
    return quantized_list


def reverse_quantize(
    weight: List[int],
    clipping_range: Optional[float] = None,
    target_range: Optional[int] = None,
) -> List[float]:
    """
    Reverse quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776
    """
    if clipping_range is None:
        clipping_range = CLIPPING_RANGE
    if target_range is None:
        target_range = TARGET_RANGE
    f = np.vectorize(
        lambda x: (x) / target_range * (2 * clipping_range) - clipping_range
    )
    weight = np.array(weight)
    reverse_quantized_list = f(weight.astype(float))
    return reverse_quantized_list


def getprimeover(bits):
    """Returns a prime number with specific number of bits"""
    random.seed(10)
    randfunc = random.SystemRandom()
    r = gmpy2.mpz(randfunc.getrandbits(bits))
    r = gmpy2.bit_set(r, bits - 1)
    return gmpy2.next_prime(r)


def invert(a, b):
    """Finds the invers of a mod b"""
    s = gmpy2.invert(a, b)
    # according to documentation, gmpy2.invert might return 0 on
    # non-invertible element, although it seems to actually raise an
    # exception; for consistency, we always raise the exception
    if s == 0:
        raise ZeroDivisionError("invert() no inverse exists")
    return s


def powmod(a, b, c):
    """Computes a to the power of b mod c"""

    if a == 1:
        return 1
    return gmpy2.powmod(a, b, c)
