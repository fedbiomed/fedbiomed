# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import json

from typing import List, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers, SAParameters
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.logger import logger

from ._config_utils import SHARE_DIR


def matching_parties_dh(context: dict, parties: list) -> bool:
    """Check if parties of given context are compatible with the parties
        of a secagg Diffie Hellman element.

        Args:
            context: context to be compared with the secagg servkey element parties
            parties: the secagg servkey element parties

        Returns:
            True if this context can be used with this element, False if not.
        """
    # Need to ensure that:
    # - no check on first party (no cryptographic material attached to the researcher).
    #   The context is established for a given experiment, thus a given researcher but this should
    #   be tested prior to this call.
    # - existing element was established for the same node parties or a superset of the node parties
    #   (order can differ, as nodes are ordered by the cipher code)
    #
    # eg: [ 'un', 'deux', 'trois' ] parties compatible with [ 'un', 'trois', 'deux' ] context
    # but not with [ 'deux', 'un', 'trois' ]
    # eg: [ 'un', 'deux', 'trois' ] parties compatible with [ 'un', 'trois', 'quatre', 'deux' ] context
    # but not with [ 'un', 'deux', 'quatre' ]
    return (
        # Commented tests can be assumed from calling functions
        #
        # isinstance(context, dict) and
        # 'parties' in context and
        # isinstance(context['parties'], list) and
        # len(context['parties']) >= 1 and
        # isinstance(parties, list) and
        set(parties[1:]).issubset(set(context['parties'][1:])))


def matching_parties_servkey(context: dict, parties: list) -> bool:
    """Check if parties of given context are compatible with the parties
        of a secagg servkey element.

        Args:
            context: context to be compared with the secagg servkey element parties
            parties: the secagg servkey element parties

        Returns:
            True if this context can be used with this element, False if not.
        """
    # Need to ensure that:
    # - existing element was established for the same parties
    # - first party needs to be the same for both
    # - set of other parties needs to be the same for both (order can differ)
    #
    # eg: [ 'un', 'deux', 'trois' ] compatible with [ 'un', 'trois', 'deux' ]
    # but not with [ 'deux', 'un', 'trois' ]
    return (
        # Commented tests can be assumed from calling functions
        #
        # isinstance(context, dict) and
        # 'parties' in context and
        # isinstance(context['parties'], list) and
        # len(context['parties']) >= 1 and
        # isinstance(parties, list) and
        parties[0] == context['parties'][0] and
        set(parties[1:]) == set(context['parties'][1:]))


def quantize(
    weights: List[float],
    clipping_range: Union[int, None] = None,
    target_range: int = SAParameters.TARGET_RANGE,
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
        clipping_range = SAParameters.CLIPPING_RANGE

    _check_clipping_range(weights, clipping_range)

    # CAVEAT: ensure to be converting from `float` to `uint64` (no intermediate `int64`)
    # Process ensures to compute an `int`` in the range [0, target_range -1]
    # This enables to use at most 2**64 as target_range (max value of `uint` - 1)
    f = np.vectorize(
        lambda x: min(
            target_range - 1,
            (sorted((-clipping_range, x, clipping_range))[1] +
            clipping_range) *
            target_range / (2 * clipping_range),
        ),
        otypes=[np.uint64]
    )
    quantized_list = f(weights)

    return quantized_list.tolist()


def multiply(xs: List[int], k: int) -> List[int]:
    """
    Multiplies a list of integers by a constant

    Args:
        xs: List of integers
        k: Constant to multiply by

    Returns:
        List of multiplied integers
    """
    # Quicker than converting to/from numpy
    return [e * k for e in xs]


def divide(xs: List[int], k: int) -> List[int]:
    """
    Divides a list of integers by a constant

    Args:
        xs: List of integers
        k: Constant to divide by

    Returns:
        List of divided integers
    """
    # Quicker than converting to/from numpy
    return [e / k for e in xs]


def reverse_quantize(
    weights: List[int],
    clipping_range: Union[int, None] = None,
    target_range: int = SAParameters.TARGET_RANGE,
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
        clipping_range = SAParameters.CLIPPING_RANGE

    # CAVEAT: there should not be any weight received that does not fit in `uint64`
    max_val = np.iinfo(np.uint64).max
    if any([v > max_val or v < 0 for v in weights]):
        raise FedbiomedSecaggCrypterError(
            f"{ErrorNumbers.FB624.value}: Cannot reverse quantize, received values exceed maximum number"
        )

    max_range = clipping_range
    min_range = -clipping_range
    step_size = (max_range - min_range) / (target_range - 1)
    # Compute as input type (`np.uint64` then convert to `np.float64`)
    f = np.vectorize(
        lambda x: (min_range + step_size * x),
        otypes=[np.float64]
    )

    # TODO: we could check that received values are in the range
    weights = np.array(weights, dtype=np.uint64)
    reverse_quantized_list = f(weights)

    return reverse_quantized_list.tolist()


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


def get_default_biprime():
    """Gets default biprime"""
    biprime = os.path.join(
        SHARE_DIR, "envs", "common", "default_biprimes", "biprime0.json")
    with open(biprime, '+r', encoding="UTF-8") as json_file:
        biprime = json.load(json_file)

    return biprime["biprime"]
