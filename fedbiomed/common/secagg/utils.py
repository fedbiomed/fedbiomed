import numpy as np

from typing import List, Union

from fedbiomed.common.constants import VEParameters
from fedbiomed.common.logger import logger


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
            "There are some numbers in the local vector that exceeds clipping range. Please increase the "
            "clipping range to account for value")


def quantize(
    weights: List[float],
    clipping_range: Union[int, None] = None,
    target_range: int = VEParameters.TARGET_RANGE,
) -> List[int]:
    """Quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776

    This function returns a vector in the range [0, target_range-1].

    Args:
        weights: List of model weight values
        clipping_range: Clipping range
        target_range: Target range

    Returns:
        Quantized model weights as numpy array.

    """
    if clipping_range is None:
        clipping_range = VEParameters.CLIPPING_RANGE

    _check_clipping_range(weights, clipping_range)

    f = np.vectorize(
        lambda x: min(
            target_range - 1,
            (sorted((-clipping_range, x, clipping_range))[1] + clipping_range) *
            target_range /
            (2 * clipping_range),
        )
    )
    quantized_list = f(weights).astype(int)

    return quantized_list.tolist()


def multiply(xs, k):
    """
    Multiplies a list of integers by a constant

    Args:
        xs: List of integers
        k: Constant to multiply by

    Returns:
        List of multiplied integers
    """
    xs = np.array(xs, dtype=np.uint32)
    return (xs * k).tolist()


def divide(xs, k):
    """
    Divides a list of integers by a constant

    Args:
        xs: List of integers
        k: Constant to divide by

    Returns:
        List of divided integers
    """
    xs = np.array(xs, dtype=np.uint32)
    return (xs / k).tolist()


def reverse_quantize(
    weights: List[int],
    clipping_range: Union[int, None] = None,
    target_range: int = VEParameters.TARGET_RANGE,
) -> List[float]:
    """Reverse quantization step implemented by: https://dl.acm.org/doi/pdf/10.1145/3488659.3493776

     Args:
        weights: List of quantized model weights
        clipping_range: Clipping range used for quantization
        target_range: Target range used for quantization

    Returns:
        Reversed quantized model weights as numpy array.
    """

    if clipping_range is None:
        clipping_range = VEParameters.CLIPPING_RANGE
    max_range = clipping_range
    min_range = -clipping_range
    step_size = (max_range - min_range) / (target_range - 1)
    f = np.vectorize(
        lambda x: (min_range + step_size * x)
    )

    weights = np.array(weights)
    reverse_quantized_list = f(weights.astype(float))

    return reverse_quantized_list.tolist()


def apply_average(
        params: List[int],
        total_weight: int
) -> List:
    """Divides parameters with total weight

    Args:
        params: List of parameters
        total_weight: Total weight to divide

    Returns:
        List of averaged parameters
    """
    return divide(params, total_weight)


def apply_weighing(
        params: List[int],
        weight: int,
) -> List[int]:
    """Multiplies parameters with weight

    Args:
        params: List of parameters
        weight: Weight to multiply

    Returns:
        List of weighted parameters
    """
    return multiply(params, weight)
