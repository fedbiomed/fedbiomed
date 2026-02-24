# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set, Type, Union

from fedbiomed.common.constants import Stats
from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.exceptions import FedbiomedError

from ._base import Accumulator
from ._scalar_1d import (
    CountAccumulator,
    HistogramAccumulator,
    MaxAccumulator,
    MeanAccumulator,
    MinAccumulator,
    QuantileAccumulator,
    VarianceAccumulator,
)

# --- Registration of Statistics ---
_REGISTRY_STATS = [
    # ===== ROW Primitives (no aggregate_channels) =====
    {
        "name": "count",
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": CountAccumulator,
    },
    {
        "name": "min",
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": MinAccumulator,
    },
    {
        "name": "max",
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": MaxAccumulator,
    },
    {
        "name": "mean",
        "dependencies": "count",
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": MeanAccumulator,
    },
    {
        "name": "variance",
        "dependencies": ["mean", "count"],
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": VarianceAccumulator,
    },
    # ===== Complex stats (shared or type-specific) =====
    {
        "name": "histogram",
        "required_args": {"bin_edges"},
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": HistogramAccumulator,
        "is_vectorizable": False,
    },
    {
        "name": "quantile",
        "required_args": {"quantiles"},
        "valid_for": DatasetElementType.ROW,
        "accumulator_class": QuantileAccumulator,
        "is_vectorizable": False,
        "uses_buffer": True,
    },
    # ===== IMAGE Primitives (with aggregate_channels support) =====
    # {
    #     "name": "count",
    #     "valid_for": DatasetElementType.IMAGE,
    #     "accumulator_class": CountAccumulator,
    #     "optional_args": {"aggregate_channels"},
    # },
    # {
    #     "name": "min",
    #     "valid_for": DatasetElementType.IMAGE,
    #     "accumulator_class": MinAccumulator,
    #     "optional_args": {"aggregate_channels"},
    # },
    # {
    #     "name": "max",
    #     "valid_for": DatasetElementType.IMAGE,
    #     "accumulator_class": MaxAccumulator,
    #     "optional_args": {"aggregate_channels"},
    # },
    # {
    #     "name": "mean",
    #     "dependencies": "count",
    #     "valid_for": DatasetElementType.IMAGE,
    #     "accumulator_class": MeanAccumulator,
    #     "optional_args": {"aggregate_channels"},
    # },
    # {
    #     "name": "variance",
    #     "dependencies": ["mean", "count"],
    #     "valid_for": DatasetElementType.IMAGE,
    #     "accumulator_class": VarianceAccumulator,
    #     "optional_args": {"aggregate_channels"},
    # },
]


@dataclass
class StatConfig:
    """Configuration for a statistic for a specific dataset element type.

    Attributes:
        accumulator_class: The accumulator class to use for this statistic.
        dependencies: Set of dependency names required to compute this statistic.
        required_args: Set of argument names that must be provided by the user.
        optional_args: Set of optional argument names.
        is_descriptor: If True, descriptor is generated from image (IMAGE only).
        is_vectorizable: If True, computation can be applied across multiple columns at once (ROW only).
        uses_buffer: If True, n_samples is passed by the orchestrator as 'buffer_size'. Cannot be set by the user.
    """

    accumulator_class: Type["Accumulator"]
    dependencies: Set[str] = field(default_factory=set)
    required_args: Set[str] = field(default_factory=set)
    optional_args: Set[str] = field(default_factory=set)
    is_descriptor: bool = False
    is_vectorizable: bool = False
    uses_buffer: bool = False


class AnalyticsRegistry:
    """Central registry for analytics statistics definitions.

    Internal Storage Structure:
        DatasetElementType -> StatName -> StatConfig
    """

    # Storage: DatasetElementType -> StatName -> StatConfig
    _registry: Dict[DatasetElementType, Dict[str, StatConfig]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        valid_for: Union[Iterable[DatasetElementType], DatasetElementType],
        accumulator_class: Type["Accumulator"],
        dependencies: Optional[Union[str, Set[str], Iterable[str]]] = None,
        required_args: Optional[Union[str, Set[str], Iterable[str]]] = None,
        optional_args: Optional[Union[str, Set[str], Iterable[str]]] = None,
        is_descriptor: bool = False,
        is_vectorizable: bool = True,
        uses_buffer: bool = False,
    ):
        """Registers a statistic configuration.

        Args:
            name: The name of the statistic to register.
            valid_for: The set, list, or single dataset element type this config applies to.
            accumulator_class: The accumulator class to use for this statistic (required).
            dependencies: Set of dependency names, or single name, required to compute this statistic.
            required_args: Set of argument names, or single name, that must be provided by the user.
            optional_args: Set of optional argument names, or single name.
            is_descriptor: If True, indicates a descriptor is generated from an image (IMAGE only).
            is_vectorizable: If True, computation can be applied across multiple columns at once (ROW only).
            uses_buffer: If True, 'n_samples' will be passed to the accumulator as 'buffer_size' by the orchestrator.

        Raises:
            FedbiomedError: If argument sets overlap or if the statistic is already registered for a target dataset element type.
        """
        if not any(name == stat.value for stat in Stats):
            raise FedbiomedError(f"Stat name: '{name}', is not defined in Stats enum.")

        # Normalize valid_for to set
        if isinstance(valid_for, DatasetElementType):
            element_types = {valid_for}
        else:
            element_types = set(valid_for)

        dependencies_set = cls._normalize(dependencies)
        required_args_set = cls._normalize(required_args)
        optional_args_set = cls._normalize(optional_args)

        for et in element_types:
            cls._registry.setdefault(et, {})

            # Check for overlaps
            if name in cls._registry[et]:
                raise FedbiomedError(
                    f"Statistic '{name}' is already registered for type: {et.name}. "
                    "Cannot overwrite existing registration."
                )

            # Handle is_descriptor flag intelligently (IMAGE only)
            current_is_descriptor = is_descriptor
            if current_is_descriptor and et != DatasetElementType.IMAGE:
                # If explicitly asked for multiple types, just disable descriptor for non-IMAGE
                current_is_descriptor = False

            # Handle is_vectorizable flag intelligently (ROW only)
            current_is_vectorizable = is_vectorizable
            if current_is_vectorizable and et != DatasetElementType.ROW:
                # If explicitly asked for multiple types, just disable vectorization for non-ROW
                current_is_vectorizable = False

            # Create a new config instance for each type to avoid shared references
            config = StatConfig(
                accumulator_class=accumulator_class,
                dependencies=dependencies_set.copy(),
                required_args=required_args_set.copy(),
                optional_args=optional_args_set.copy(),
                is_descriptor=current_is_descriptor,
                is_vectorizable=current_is_vectorizable,
                uses_buffer=uses_buffer,
            )
            # Parent dictionary guaranteed to exist by previous loop
            cls._registry[et][name] = config

    @classmethod
    def _normalize(cls, value: Optional[Union[str, Iterable[str]]]) -> Set[str]:
        """Helper to normalize input into a set of strings."""
        return {value} if isinstance(value, str) else set(value or [])

    @classmethod
    def get_accumulator_class(
        cls, stat: str, element_type: DatasetElementType
    ) -> Optional[Type["Accumulator"]]:
        """Retrieve the accumulator class for a statistic and type.

        Args:
            stat: The name of the statistic.
            element_type: The dataset element type.

        Returns:
            Accumulator class if registered, None otherwise.
        """
        type_registry = cls._registry.get(element_type, {})
        config = type_registry.get(stat)
        return config.accumulator_class if config else None

    @classmethod
    def _is_known_stat(cls, name: str) -> bool:
        """Checks if a statistic is registered for ANY type.

        Args:
            name: The name of the statistic.

        Returns:
            True if registered anywhere, False otherwise.
        """
        for stats in cls._registry.values():
            if name in stats:
                return True
        return False

    @classmethod
    def get(cls, name: str) -> Optional[Dict[DatasetElementType, StatConfig]]:
        """Retrieves a statistic configuration map by name.

        Args:
            name: The name of the statistic.

        Returns:
            A dictionary mapping DatasetElementType to StatConfig if found for any type, None otherwise.
        """
        results = {}
        for et, stats in cls._registry.items():
            if name in stats:
                results[et] = stats[name]
        return results if results else None

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
        type_registry = cls._registry.get(element_type, {})

        if stat not in type_registry:
            # It's not valid for this type. Is it valid for ANY type?
            if not cls._is_known_stat(stat):
                raise FedbiomedError(f"Unknown statistic: {stat}")
            else:
                raise FedbiomedError(
                    f"Statistic '{stat}' is not valid for type {element_type.name}"
                )

        config = type_registry[stat]
        provided = set(args.keys())
        missing = config.required_args - provided
        if missing:
            raise FedbiomedError(f"Statistic '{stat}' missing required args: {missing}")

        # Check for unexpected arguments
        allowed = config.required_args | config.optional_args
        unexpected = provided - allowed
        if unexpected:
            raise FedbiomedError(
                f"Statistic '{stat}' received unexpected args: {unexpected}"
            )

    @classmethod
    def get_dependencies(
        cls, stats: Union[str, Iterable[str]], element_type: DatasetElementType
    ) -> Set[str]:
        """Retrieves all dependencies required for one or more statistics.

        It recursively traverses the dependencies and returns the full set of
        required statistics. The input statistics themselves are not included
        in the result unless they appear as dependencies of other inputs.

        Args:
            stats: Single statistic name or iterable of names.
            element_type: The dataset element type.

        Returns:
            A set of dependency names.
        """
        if isinstance(stats, str):
            stats_list = [stats]
        else:
            stats_list = list(stats)

        dependencies = set()
        queue = list(stats_list)
        visited = set(stats_list)

        while queue:
            current = queue.pop(0)

            # Retrieve immediate dependencies
            type_registry = cls._registry.get(element_type, {})
            config = type_registry.get(current)
            immediate_deps = config.dependencies if config else set()

            for dep in immediate_deps:
                # Always add to dependencies set
                dependencies.add(dep)

                # Queue for traversal if not already visited
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        return dependencies

    @classmethod
    def check_stat_compatibility(
        cls, stat: str, element_type: DatasetElementType
    ) -> bool:
        """Checks if a statistic is valid for a given dataset element type.

        This check verifies compatibility. If the statistic is completely unknown (not registered
        for any type), it raises an error.

        Args:
            stat: The name of the statistic.
            element_type: The type of dataset element (e.g. ROW, IMAGE).

        Returns:
            True if valid, False otherwise.

        Raises:
            FedbiomedError: If the statistic is unknown (globally).
        """
        if stat in cls._registry.get(element_type, {}):
            return True

        if not cls._is_known_stat(stat):
            raise FedbiomedError(f"Unknown statistic: {stat}")

        return False

    @classmethod
    def get_roots(
        cls, stats: Iterable[str], element_type: DatasetElementType
    ) -> Set[str]:
        """Identifies the 'root' statistics from a list of requested statistics.

        A statistic is a root if it:
            1. Is explicitly requested (in the input stats list).
            2. Is NOT a dependency (transitive) of any OTHER statistic in the requested list.

        Args:
            stats: The list of statistics to filter.
            element_type: The dataset element type context.

        Returns:
            A subset of 'stats' that are roots.
        """
        requested = set(stats)
        implied_dependencies = set()

        for s in requested:
            # Get full dependencies for this specific stat
            stat_deps = cls.get_dependencies(s, element_type)
            implied_dependencies.update(stat_deps)

        # Roots are items in the requested list that are NOT implied by others in the list
        return requested - implied_dependencies

    @classmethod
    def initialize(cls) -> None:
        """Registers the default statistics configuration defined in _REGISTRY_STATS."""
        for stat_config in _REGISTRY_STATS:
            cls.register(**stat_config)


AnalyticsRegistry.initialize()
