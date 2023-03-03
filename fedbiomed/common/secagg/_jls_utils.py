
from typing import List

import gmpy2
import numpy as np
from gmpy2 import mpz

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import VEParameters


def _check_clipping_range(
        values: List[float],
        clipping_range: float
) -> None:
    """Checks clipping range for quantization

    Args:
        values: Values to check whether clipping range is exceed from both direction
        clipping_range: Clipping range

    """
    state = False

    for x in values:
        if x < -clipping_range or x > clipping_range:
            state = True

    if state:
        logger.info(
            f"There are some numbers in the local vector that exceeds clipping range. Please increase the "
            f"clipping range to account for value")


def quantize(
    weights: List[float],
    clipping_range: int = VEParameters.CLIPPING_RANGE,
    target_range: int = VEParameters.TARGET_RANGE,
) -> np.ndarray:
    """Quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776

    This function returns a vector in the range [0, target_range-1].

    Args:
        weights: List of model weight values
        clipping_range: Clipping range
        target_range: Target range

    Returns:
        Quantized model weights as numpy array.

    """

    # Check clipping range
    _check_clipping_range(weights, clipping_range)

    f = np.vectorize(
        lambda x: min(
            target_range - 1,
            (sorted((-clipping_range, x, clipping_range))[1] + clipping_range)
            * target_range
            / (2 * clipping_range),
        )
    )
    quantized_list = f(weights).astype(int)

    return quantized_list


def reverse_quantize(
    weights: List[int],
    clipping_range: float = VEParameters.CLIPPING_RANGE,
    target_range: int = VEParameters.TARGET_RANGE,
) -> np.ndarray:
    """Reverse quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776

     Args:
        weights: List of quantized model weights
        clipping_range: Clipping range used for quantization
        target_range: Target range used for quantization

    Returns:
        Reversed quantized model weights as numpy array.
    """

    f = np.vectorize(
        lambda x: x / target_range * (2 * clipping_range) - clipping_range
    )

    weights = np.array(weights)
    reverse_quantized_list = f(weights.astype(float))

    return reverse_quantized_list


def invert(
        a: mpz,
        b: mpz
) -> mpz:
    """Finds the inverts of a mod b"""

    s = gmpy2.invert(a, b)
    # according to documentation, gmpy2.invert might return 0 on
    # non-invertible element, although it seems to actually raise an
    # exception; for consistency, we always raise the exception
    if s == 0:
        raise ZeroDivisionError("invert() no inverse exists")
    return s


def powmod(
        a: mpz,
        b: mpz,
        c: mpz
) -> mpz:
    """Computes a to the power of b mod c"""
    if a == 1:
        return 1
    return gmpy2.powmod(a, b, c)
