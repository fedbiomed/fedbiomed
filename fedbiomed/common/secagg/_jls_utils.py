import random
import gmpy2
import numpy as np

from typing import List, Optional

CLIPPING_RANGE = 3
TARGET_RANGE = 10000


def _check_clipping_range(values: List[float], clipping_range: float):
    """Checks clipping range for quantization

    Args:
        values: Values to check whether clipping range is exceed from both direction
        clipping_range: Clipping range

    """
    for x in values:
        if x < -clipping_range or x > clipping_range:
            raise ValueError(
                f"There are some numbers in the local vector that exceeds clipping range. Please increase the "
                f"clipping range to account for value {x}",
            )


def quantize(
    weights: List[float],
    clipping_range: Optional[float] = None,
    target_range: Optional[int] = None,
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
    # Assign defaults
    if clipping_range is None:
        clipping_range = CLIPPING_RANGE
    if target_range is None:
        target_range = TARGET_RANGE

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
    clipping_range: Optional[float] = None,
    target_range: Optional[int] = None,
) -> np.ndarray:
    """Reverse quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776

     Args:
        weights: List of quantized model weights
        clipping_range: Clipping range used for quantization
        target_range: Target range used for quantization

    Returns:
        Reversed quantized model weights as numpy array.
    """
    if clipping_range is None:
        clipping_range = CLIPPING_RANGE
    if target_range is None:
        target_range = TARGET_RANGE

    f = np.vectorize(
        lambda x: (x) / target_range * (2 * clipping_range) - clipping_range
    )

    weights = np.array(weights)
    reverse_quantized_list = f(weights.astype(float))

    return reverse_quantized_list


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
