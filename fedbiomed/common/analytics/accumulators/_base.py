# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

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

    def __init__(self, children: List[Accumulator], is_tuple: bool = True):
        self.children = children
        self.is_tuple = is_tuple

    def update(self, value: Union[List, Tuple]) -> None:
        if not isinstance(value, (list, tuple)):
            raise FedbiomedError(
                f"SequenceAccumulator expected sequence, got {type(value)}"
            )

        for i, child in enumerate(self.children):
            if i < len(value):
                child.update(value[i])

    def finalize(self) -> Union[List, Tuple]:
        results = [child.finalize() for child in self.children]
        if self.is_tuple:
            return tuple(results)
        return results


class SkipAccumulator(Accumulator):
    """Accumulator that produces no output; used when a sequence position is explicitly skipped."""

    def update(self, value: Any) -> None:
        pass

    def finalize(self) -> None:
        return None
