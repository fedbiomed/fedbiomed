# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from fedbiomed.common.dataset_types import DatasetElementType


@dataclass
class StatMetadata:
    """Metadata defining validation and behavior for a statistic."""

    name: str
    valid_for: Set[DatasetElementType]  # Must be explicitly provided
    dependencies: Set[str] = field(default_factory=set)
    required_args: Set[str] = field(default_factory=set)
    optional_args: Set[str] = field(default_factory=set)
    is_descriptor: bool = (
        False  # If true, treated as scalar descriptor (Image) rather than pixel-wise
    )


class AnalyticsRegistry:
    """Central registry for analytics statistics definitions."""

    _registry: Dict[str, StatMetadata] = {}

    @classmethod
    def register(cls, meta: StatMetadata):
        """Registers a statistic metadata.

        Args:
           meta: The metadata of the statistic to register.
        """
        cls._registry[meta.name] = meta

    @classmethod
    def get(cls, name: str) -> Optional[StatMetadata]:
        """Retrieves a statistic metadata by name.

        Args:
            name: The name of the statistic.

        Returns:
            The metadata if found, None otherwise.
        """
        return cls._registry.get(name)

    @classmethod
    def validate_args(cls, stat: str, args: Dict) -> None:
        """Validates arguments for a statistic against its metadata.

        Args:
            stat: Map of the statistic.
            args: Map of arguments provided for the statistic.

        Raises:
            ValueError: If the statistic is unknown or if required arguments are missing.
        """
        meta = cls.get(stat)
        if not meta:
            raise ValueError(f"Unknown statistic: {stat}")

        provided = set(args.keys())
        missing = meta.required_args - provided
        if missing:
            raise ValueError(f"Statistic '{stat}' missing required args: {missing}")

    @classmethod
    def get_dependencies(cls, stat: str) -> Set[str]:
        """Retrieves all dependencies (direct and valid recursive) for a statistic.

        Args:
            stat: The name of the statistic.

        Returns:
            A set of dependency names.
        """
        meta = cls.get(stat)
        if not meta:
            return set()

        deps = set(meta.dependencies)
        # Recursive dependency check
        for dep in meta.dependencies:
            deps.update(cls.get_dependencies(dep))
        return deps

    @classmethod
    def is_valid_for_type(cls, stat: str, element_type: DatasetElementType) -> bool:
        """Checks if a statistic is valid for a given dataset element type.

        Args:
            stat: The name of the statistic.
            element_type: The type of dataset element (e.g. ROW, IMAGE).

        Returns:
            True if valid, False otherwise.

        Raises:
            ValueError: If the statistic is unknown.
        """
        meta = cls.get(stat)
        if not meta:
            raise ValueError(f"Unknown statistic: {stat}")

        return element_type in meta.valid_for


# --- Registration of Standard Statistics ---

# Numeric / Basic
AnalyticsRegistry.register(
    StatMetadata(
        name="count", valid_for={DatasetElementType.ROW, DatasetElementType.IMAGE}
    )
)

AnalyticsRegistry.register(
    StatMetadata(
        name="mean",
        dependencies={"count"},
        valid_for={DatasetElementType.ROW, DatasetElementType.IMAGE},
    )
)

AnalyticsRegistry.register(
    StatMetadata(
        name="std",
        dependencies={"mean", "count"},
        valid_for={DatasetElementType.ROW, DatasetElementType.IMAGE},
    )
)

AnalyticsRegistry.register(
    StatMetadata(
        name="min", valid_for={DatasetElementType.ROW, DatasetElementType.IMAGE}
    )
)

AnalyticsRegistry.register(
    StatMetadata(
        name="max", valid_for={DatasetElementType.ROW, DatasetElementType.IMAGE}
    )
)

AnalyticsRegistry.register(StatMetadata(name="sum", valid_for={DatasetElementType.ROW}))

# Complex / Tabular
AnalyticsRegistry.register(
    StatMetadata(
        name="histogram",
        required_args={"bin_edges"},
        optional_args={"bins"},  # bin_edges overrides bins usually
        valid_for={DatasetElementType.ROW},
    )
)

AnalyticsRegistry.register(
    StatMetadata(
        name="quantiles", required_args={"q"}, valid_for={DatasetElementType.ROW}
    )
)

# Image Specific
AnalyticsRegistry.register(
    StatMetadata(name="shape", valid_for={DatasetElementType.IMAGE}, is_descriptor=True)
)

AnalyticsRegistry.register(
    StatMetadata(
        name="intensity_mean",  # Global intensity aggregation
        dependencies={"mean"},  # Uses base mean logic internally? Or custom?
        # Actually ImageAccumulator uses 'intensity' subscale which uses 'mean'.
        # But from orchestrator POV, it's a stat requested on image.
        valid_for={DatasetElementType.IMAGE},
        is_descriptor=True,
    )
)

AnalyticsRegistry.register(
    StatMetadata(
        name="intensity_std", valid_for={DatasetElementType.IMAGE}, is_descriptor=True
    )
)
