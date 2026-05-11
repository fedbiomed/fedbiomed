# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import numpy as np

from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.logger import logger

from ._base import Accumulator
from ._registry import AnalyticsRegistry


class ImageAccumulator(Accumulator):
    """Accumulator for image-type dataset elements.

    Holds one accumulator per requested statistic and dispatches each image
    sample to all of them. Unlike RowAccumulator, there is no column or
    vectorization concept; each stat accumulator receives the full image array.

    Image-specific logic (e.g. channel aggregation) is delegated to the
    individual stat accumulator classes, which will be defined in _operations.py.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary from the orchestrator.
                Expected format::

                    {
                        "type": DatasetElementType.IMAGE,
                        "stats": {
                            "mean": {},
                            "histogram": {"bin_edges": [...]},
                        }
                    }
        """
        self.stats_config: Dict[str, Any] = config.get("stats", {})
        self.accumulators: List[Accumulator] = []

        # Group stats by args so accumulators sharing the same config are built together
        groups: List[tuple] = []
        for stat, stat_args in self.stats_config.items():
            for group_args, group_stats in groups:
                if group_args == stat_args:
                    group_stats.append(stat)
                    break
            else:
                groups.append((stat_args, [stat]))

        for stat_args, stats in groups:
            accumulator_cls_list = AnalyticsRegistry.get_accumulators(
                stats, DatasetElementType.IMAGE
            )
            for acc_cls in accumulator_cls_list:
                self.accumulators.append(acc_cls(**stat_args))

        logger.debug("ImageAccumulator initialized")

    def update(self, value: np.ndarray) -> None:
        """Update all stat accumulators with a new image sample.

        Args:
            value: Image array of arbitrary shape (e.g. (H, W) or (C, H, W)).
        """
        value = np.asarray(value)
        for acc in self.accumulators:
            acc.update(value)

    def finalize(self) -> Dict[str, Any]:
        """Return finalized statistics for this image element.

        Returns:
            A merged dictionary of all stat accumulator results.
        """
        result = {}
        for acc in self.accumulators:
            result.update(acc.finalize())
        return result
