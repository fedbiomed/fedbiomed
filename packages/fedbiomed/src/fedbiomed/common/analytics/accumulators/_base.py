# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from fedbiomed.common.constants import FedbiomedError


class Accumulator(ABC):
    """Abstract base class for streaming statistics accumulators."""

    @abstractmethod
    def update(self, value: Any) -> None:
        """Update the running statistics with a new value."""
        pass

    @abstractmethod
    def finalize(self) -> Any:
        """Return the final computed statistics."""
        pass


# --- Structural Accumulators ---


class DictAccumulator(Accumulator):
    """Accumulator for dictionary structures."""

    def __init__(self, children: Dict[str, Accumulator]):
        self.children = children

    def update(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise FedbiomedError(f"DictAccumulator expected dict, got {type(value)}")

        for key, child in self.children.items():
            if key in value:
                child.update(value[key])

    def finalize(self) -> Dict[str, Any]:
        return {key: child.finalize() for key, child in self.children.items()}


class SequenceAccumulator(Accumulator):
    """Accumulator for sequence structures (lists, tuples)."""

    def __init__(
        self,
        children: List[Accumulator],
        indices: Optional[List[int]] = None,
    ):
        self.children = children
        self.indices = indices if indices is not None else list(range(len(children)))

    def update(self, value: Union[List, Tuple]) -> None:
        if not isinstance(value, (list, tuple)):
            raise FedbiomedError(
                f"SequenceAccumulator expected sequence, got {type(value)}"
            )

        for idx, child in zip(self.indices, self.children, strict=True):
            if idx >= len(value):
                raise FedbiomedError(
                    f"SequenceAccumulator: index {idx} out of range for value of length {len(value)}"
                )
            child.update(value[idx])

    def finalize(self) -> Any:
        results = [child.finalize() for child in self.children]
        if len(results) == 1:
            return results[0]
        return results
