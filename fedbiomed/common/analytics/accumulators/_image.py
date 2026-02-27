# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np

from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.exceptions import FedbiomedError
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
        self.accumulators: Dict[str, Accumulator] = {}

        logger.info(
            f"Initializing ImageAccumulator with stats: {list(self.stats_config.keys())}"
        )

        for stat, stat_args in self.stats_config.items():
            accumulator_class = AnalyticsRegistry.get_accumulator_class(
                stat, DatasetElementType.IMAGE
            )
            if accumulator_class is None:
                raise FedbiomedError(
                    f"No accumulator registered for stat '{stat}' of type IMAGE."
                )
            self.accumulators[stat] = accumulator_class(**stat_args)

    def update(self, value: np.ndarray) -> None:
        """Update all stat accumulators with a new image sample.

        Args:
            value: Image array of arbitrary shape (e.g. H×W or C×H×W).
        """
        value = np.asarray(value)
        for acc in self.accumulators.values():
            acc.update(value)

    def finalize(self) -> Dict[str, Any]:
        """Return finalized statistics for this image element.

        Returns:
            A dictionary mapping stat names to their finalized values.
        """
        return {stat: acc.finalize() for stat, acc in self.accumulators.items()}
