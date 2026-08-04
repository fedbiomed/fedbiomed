# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np


def flatten_fa_output(output: dict) -> tuple[list[float], list]:
    """Depth-first traversal of an FA stats dict, collecting every numeric scalar.

    Each numeric scalar (int or float, including values inside nested lists) is
    appended to a flat list.  Its position in the original structure is recorded
    as a key-path — a list whose elements are either string dict keys or integer
    list indices.  Passing the two return values to ``unflatten_fa_output``
    reconstructs the original dict exactly.

    Non-numeric leaf values (e.g. strings) are silently skipped; they are not
    encryptable and should not appear in FA sufficient-statistic output.

    Args:
        output: FA stats dict produced by ``dataset.compute_stats``.

    Returns:
        As flat, a list of floats with every numeric scalar in depth-first order.
        As schema, a parallel list of key-paths (each a list of str/int keys)
            that describes where each value came from.
    """
    flat: list[float] = []
    schema: list = []

    def _traverse(obj: Any, path: list) -> None:
        if isinstance(obj, dict):
            # Sort keys for a node-independent, deterministic layout
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
                _traverse(v, path + [k])
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _traverse(v, path + [i])
        elif isinstance(obj, (int, float, np.floating, np.integer)) and not isinstance(
            obj, (bool, np.bool_)
        ):
            flat.append(float(obj))
            schema.append(path)

    _traverse(output, [])
    return flat, schema


def unflatten_fa_output(flat: list, schema: list) -> dict:
    """Reconstruct an FA stats dict from a flat value list and its schema.

    This is the inverse of ``flatten_fa_output``.  Each value in *flat* is
    placed back at the position described by the corresponding entry in *schema*.

    Args:
        flat: Flat list of numeric values (floats after decryption).
        schema: List of key-paths produced by ``flatten_fa_output``.

    Returns:
        Reconstructed stats dict.
    """
    result: dict = {}

    def _set_at(path: list, value: float) -> None:
        obj: Any = result
        for i, key in enumerate(path[:-1]):
            next_key = path[i + 1]
            # Determine the container type needed for the next level.
            next_is_list = isinstance(next_key, int)
            if isinstance(key, int):
                _extend_list(obj, key, next_is_list)
                obj = obj[key]
            else:
                if key not in obj:
                    obj[key] = [] if next_is_list else {}
                obj = obj[key]

        final_key = path[-1]
        if isinstance(final_key, int):
            _extend_list(obj, final_key, False)
            obj[final_key] = value
        else:
            obj[final_key] = value

    def _extend_list(lst: list, index: int, fill_with_list: bool) -> None:
        """Grow *lst* so that index is valid, filling gaps with the right type."""
        while len(lst) <= index:
            lst.append([] if fill_with_list else None)

    for value, path in zip(flat, schema, strict=True):
        _set_at(path, value)

    return result
