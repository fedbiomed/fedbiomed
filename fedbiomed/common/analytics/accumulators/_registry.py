# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, List, Type, Union

from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.exceptions import FedbiomedError

from ._base import Accumulator
from ._operations import (
    CountAccumulator,
    HistogramAccumulator,
    SumAccumulator,
    SumSqAccumulator,
)


class AnalyticsRegistry:
    """Central registry for analytics statistics definitions."""

    _REGISTRY: Dict[DatasetElementType, Dict[str, List[Type[Accumulator]]]] = {
        DatasetElementType.ROW: {
            "count": [CountAccumulator],
            "sum": [SumAccumulator],
            "sum_sq": [SumSqAccumulator],
            "mean": [CountAccumulator, SumAccumulator],
            "variance": [CountAccumulator, SumAccumulator, SumSqAccumulator],
            "std": [CountAccumulator, SumAccumulator, SumSqAccumulator],
            "histogram": [HistogramAccumulator],
        },
    }

    @classmethod
    def get_accumulators(
        cls, stats: Union[str, List[str]], element_type: DatasetElementType
    ) -> List[Type[Accumulator]]:
        """Retrieve a deduplicated list of accumulator classes for one or more statistics.

        Args:
            stats: A single statistic name or a list of names.
            element_type: The dataset element type.

        Returns:
            Unique list of accumulator classes across all requested statistics.
            Empty list if none of the statistics are registered for the given type.
        """
        if isinstance(stats, str):
            stats = [stats]
        et_REGISTRY = cls._REGISTRY.get(element_type, {})
        return list(
            {
                c
                for stat in stats
                if (accumulator_list := et_REGISTRY.get(stat))
                for c in accumulator_list
            }
        )

    @classmethod
    def validate_args(
        cls, stat: str, element_type: DatasetElementType, args: Dict[str, Any]
    ) -> None:
        """Validates arguments for a statistic against its metadata for a specific type.

        Args:
            stat: Name of the statistic.
            element_type: The dataset element type (ROW/IMAGE).
            args: Map of arguments provided for the statistic.

        Raises:
            FedbiomedError: If the statistic is unknown or if required arguments are missing.
        """
        accumulators = cls.get_accumulators(stat, element_type)
        if not accumulators:
            raise FedbiomedError(
                f"Statistic '{stat}' is not valid for type {element_type.name}"
            )

        provided = set(args.keys())
        required, optional = set(), set()
        for acc_cls in accumulators:
            for name, param in inspect.signature(acc_cls.__init__).parameters.items():
                if name == "self":
                    continue
                if param.default is inspect.Parameter.empty:
                    required.add(name)
                else:
                    optional.add(name)

        missing = required - provided
        if missing:
            raise FedbiomedError(f"Statistic '{stat}' missing required args: {missing}")

        unexpected = provided - (required | optional)
        if unexpected:
            raise FedbiomedError(
                f"Statistic '{stat}' received unexpected args: {unexpected}"
            )
