# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from fedbiomed.common.analytics.accumulators import (
    Accumulator,
    AnalyticsRegistry,
    DictAccumulator,
    ImageAccumulator,
    RowAccumulator,
    SequenceAccumulator,
)
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetElementSpec,
    DatasetElementType,
    RowSpec,
)
from fedbiomed.common.exceptions import FedbiomedError

if TYPE_CHECKING:
    from fedbiomed.common.dataset import Dataset


class AnalyticsOrchestrator:
    """Orchestrates the computation of analytics over a dataset."""

    def compute_stats(
        self,
        dataset: "Dataset",
        dataset_schema: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        stats: Optional[List[str]] = None,
        stats_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Computes the requested statistics over the dataset.

        Args:
            dataset: The dataset to compute statistics for.
            dataset_schema: Selection to filter the schema (e.g. subset of columns/keys).
            stats: Default list of statistics to compute (e.g. ['mean', 'std']).
            stats_args: Specific arguments for statistics, structured matching the schema.
                        e.g. {'image': {'histogram': {'bin_edges': [...]}}}

        Returns:
            The computed statistics.

        Raises:
            FedbiomedError: If both 'stats' and 'stats_args' are empty/None, if validation
                fails, or if the dataset is missing required analytics capabilities.
        """
        # Validation: Ensure at least one of stats or stats_args is provided
        if not stats and not stats_args:
            raise FedbiomedError(
                "At least one of 'stats' or 'stats_args' must be provided."
            )

        # Analytics currently only supports datasets that return data in numpy format
        if dataset.to_format != DataReturnFormat.SKLEARN:
            raise FedbiomedError(
                f"Dataset format: '{dataset.to_format.value}' is not supported for analytics."
            )

        # Check Capability
        if not hasattr(dataset, "analytics_schema"):
            raise FedbiomedError("Dataset does not implement 'analytics_schema'.")

        # Get Schema
        schema = dataset.analytics_schema()

        # Get dataset size (needed by buffer-backed accumulators like quantile)
        n_samples = len(dataset)

        # Build & Validate Configuration
        config = self._build_and_validate_config(
            schema, dataset_schema, stats, stats_args, n_samples
        )

        # Build Accumulator Tree
        accumulator = self._create_accumulator(config)

        # Iterate and Accumulate
        for sample in dataset:
            accumulator.update(sample)

        # Finalize
        return accumulator.finalize()

    def _create_accumulator(self, config: Any) -> Accumulator:
        """Factory method to build an accumulator tree based on a configuration dict."""
        if not isinstance(config, dict) or "type" not in config:
            raise FedbiomedError("Invalid accumulator configuration.")

        # Structure Types
        if config["type"] == "dict":
            children = {
                k: self._create_accumulator(v)
                for k, v in config.get("children", {}).items()
            }
            return DictAccumulator(children)
        if config["type"] == "sequence":
            children = [
                self._create_accumulator(item) for item in config.get("children", [])
            ]
            return SequenceAccumulator(children, indices=config["indices"])

        # Leaf Types
        if config["type"] == DatasetElementType.ROW:
            return RowAccumulator(config)
        if config["type"] == DatasetElementType.IMAGE:
            return ImageAccumulator(config)

        # Unhandled types
        raise FedbiomedError(f"Unsupported accumulator type: {config['type']}")

    def _build_and_validate_config(
        self,
        schema: Any,
        subschema: Optional[Any],
        stats: Optional[List[str]],
        stats_args: Optional[Dict[str, Any]],
        n_samples: int,
    ) -> Dict[str, Any]:
        """Validates inputs and builds the configuration tree in a single pass."""
        # Dispatch based on schema type
        if isinstance(schema, dict):
            return self._handle_dict(schema, subschema, stats, stats_args, n_samples)
        if isinstance(schema, (list, tuple)):
            return self._handle_sequence(
                schema, subschema, stats, stats_args, n_samples
            )

        # Leaf Handling
        element_type = schema.type if isinstance(schema, DatasetElementSpec) else None
        if element_type == DatasetElementType.ROW:
            return self._handle_row(schema, subschema, stats, stats_args, n_samples)
        if element_type == DatasetElementType.IMAGE:
            # Image does not support subschema selection
            if subschema is not None:
                raise FedbiomedError(
                    "Subschema selection is not applicable for IMAGE type."
                )
            return self._handle_image(stats, stats_args, n_samples)

        raise FedbiomedError(f"Unsupported schema type or element type: {type(schema)}")

    def _handle_dict(
        self,
        schema: Dict,
        subschema: Optional[Union[List[Union[str, Dict[str, Any]]], Dict[str, Any]]],
        stats: Optional[List[str]],
        stats_args: Optional[Dict[str, Any]],
        n_samples: int,
    ) -> Dict[str, Any]:
        # Validate Args
        if stats_args is not None:
            if not isinstance(stats_args, dict):
                raise FedbiomedError(
                    f"Args for dict schema must be a dict. Got {type(stats_args)}."
                )
            invalid = set(stats_args.keys()) - set(schema.keys())
            if invalid:
                raise FedbiomedError(f"Args keys {invalid} not found in schema.")

        # keys_map: maps key required in output -> subschema for that key
        keys_map = {}

        if subschema is None:
            # All keys, no subschema
            for k in schema.keys():
                keys_map[k] = None
        elif isinstance(subschema, (list, tuple)):
            for item in subschema:
                key = None
                child_sub = None

                if isinstance(item, str):
                    key = item
                elif isinstance(item, dict):
                    if len(item) != 1:
                        raise FedbiomedError(
                            "Dict item in subschema list must have exactly one key."
                        )
                    key = next(iter(item))
                    child_sub = item[key]
                else:
                    raise FedbiomedError(
                        f"Invalid element type in subschema list: {type(item)}"
                    )

                if key not in schema:
                    raise FedbiomedError(f"Invalid key in subschema: {key}")
                if key in keys_map:
                    raise FedbiomedError(f"Duplicate key in subschema: {key}")

                keys_map[key] = child_sub
        elif isinstance(subschema, dict):
            invalid = set(subschema.keys()) - set(schema.keys())
            if invalid:
                raise FedbiomedError(f"Invalid keys in subschema: {invalid}")
            keys_map = subschema
        else:
            raise FedbiomedError(
                f"Subschema for dict must be list/tuple or dict. Got {type(subschema)}."
            )

        children = {}
        for k, child_sub in keys_map.items():
            child_args = stats_args.get(k) if stats_args else None
            children[k] = self._build_and_validate_config(
                schema[k], child_sub, stats, child_args, n_samples
            )

        return {"type": "dict", "children": children}

    def _handle_sequence(
        self,
        schema: Union[List, Tuple],
        subschema: Optional[Union[List, Tuple]],
        stats: Optional[List[str]],
        stats_args: Optional[Union[List, Tuple]],
        n_samples: int,
    ) -> Dict[str, Any]:
        # subschema and stats_args are indexed against active (non-None) positions only.
        active = [(orig_idx, s) for orig_idx, s in enumerate(schema) if s is not None]

        # Validate Subschema
        if subschema is not None:
            if isinstance(subschema, str):
                subschema = [subschema]
            elif not isinstance(subschema, (list, tuple)):
                raise FedbiomedError(
                    f"Subschema for sequence must be list/tuple or str. Got {type(subschema)}."
                )
            if len(subschema) != len(active):
                # Dataset schemas use the (data_schema, None) convention. Allow unwrapping for user convenience.
                if len(active) == 1:
                    subschema = [subschema]
                else:
                    raise FedbiomedError(
                        f"Subschema ({len(subschema)}) does not match schema elements "
                        f"({len(active)}). Use None at a position to exclude that element."
                    )

        # Validate Args
        if stats_args is not None:
            if not isinstance(stats_args, (list, tuple)):
                raise FedbiomedError(
                    f"Args for sequence must be list/tuple. Got {type(stats_args)}."
                )
            if len(stats_args) != len(active):
                raise FedbiomedError(
                    "Args length mismatch. Pass None to ignore elements in list/tuple."
                )

        children = []
        indices = []
        for (orig_idx, item_schema), child_sub, child_args in zip(
            active,
            subschema if subschema is not None else [None] * len(active),
            stats_args if stats_args is not None else [None] * len(active),
            strict=True,
        ):
            if subschema is not None and child_sub is None:
                continue  # User explicitly excluded this element
            children.append(
                self._build_and_validate_config(
                    item_schema, child_sub, stats, child_args, n_samples
                )
            )
            indices.append(orig_idx)

        if len(children) == 0:
            raise FedbiomedError(
                "Sequence schema produced no selectable elements. "
                "Ensure at least one schema item is non-None and not excluded by subschema."
            )
        return {
            "type": "sequence",
            "children": children,
            "indices": indices,
        }

    def _handle_row(
        self,
        schema: RowSpec,
        subschema: Optional[List[str]],
        stats: Optional[List[str]],
        stats_args: Optional[Dict[str, Any]],
        n_samples: int,
    ) -> Dict[str, Any]:
        # Validate Subschema
        if subschema is not None:
            if isinstance(subschema, str):
                subschema = [subschema]
            elif not isinstance(subschema, (list, tuple)):
                raise FedbiomedError(
                    "Subschema for ROW must be a list of columns or a single string."
                )
            invalid = set(subschema) - set(schema.columns)
            if invalid:
                raise FedbiomedError(f"Invalid columns in subschema: {invalid}")
        selected_cols = subschema if subschema is not None else schema.columns

        # Validate Args
        if stats_args is not None:
            if not isinstance(stats_args, dict):
                raise FedbiomedError("Args for ROW must be a dict.")
            invalid = set(stats_args.keys()) - set(schema.columns)
            if invalid:
                raise FedbiomedError(f"Invalid columns in args: {invalid}")

        # Compile Config
        col_configs = {}
        for col in selected_cols:
            col_args = stats_args.get(col) if stats_args else None
            col_configs[col] = self._compile_leaf_stats(
                DatasetElementType.ROW, stats, col_args, n_samples
            )

        return {
            "type": DatasetElementType.ROW,
            "conf": col_configs,
            "columns": selected_cols,  # Preserve column order for accumulators
        }

    def _handle_image(
        self,
        stats: Optional[List[str]],
        stats_args: Optional[Dict[str, Any]],
        n_samples: int,
    ) -> Dict[str, Any]:
        """Validates and builds config for IMAGE type."""
        if stats_args is not None and not isinstance(stats_args, dict):
            raise FedbiomedError("Args for IMAGE must be a dict.")

        stats_config = self._compile_leaf_stats(
            DatasetElementType.IMAGE, stats, stats_args, n_samples
        )
        return {"type": DatasetElementType.IMAGE, "stats": stats_config}

    def _compile_leaf_stats(
        self,
        element_type: DatasetElementType,
        stats: Optional[List[str]],
        stats_args: Optional[Dict[str, Any]],
        n_samples: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Compiles valid statistics configuration for a leaf node.

        Returns a flat config with requested root statistics.
        Primitives are implicitly handled by the Accumulator based on root stats.
        """
        requested_config = {}
        stats_args = stats_args or {}
        stats = stats or []

        candidates = set(stats).union(stats_args.keys())

        # Validate and filter candidates
        for stat in candidates:
            # TODO: Temporal protection to allow just approved stats to be called
            if stat not in ["count", "mean", "variance"]:
                raise FedbiomedError(f"Statistic '{stat}' is not implemented yet.")

            is_explicit = stat in stats_args
            current_args = stats_args.get(stat, {})
            if self._validate_leaf_stat(element_type, stat, current_args, is_explicit):
                requested_config[stat] = current_args

        # Inject 'buffer_size' for stats with 'uses_buffer' flag in the registry
        for stat_name in list(requested_config):
            type_map = AnalyticsRegistry.get(stat_name)
            stat_cfg = type_map.get(element_type) if type_map else None
            if stat_cfg and stat_cfg.uses_buffer:
                requested_config[stat_name] = {
                    **requested_config[stat_name],
                    "buffer_size": n_samples,
                }

        # Resolve roots and ensure consistency for requested stats
        return self._resolve_and_validate_roots(element_type, requested_config)

    def _validate_leaf_stat(
        self,
        element_type: DatasetElementType,
        stat: str,
        args: Dict[str, Any],
        is_explicit: bool,
    ) -> bool:
        """Validates a single statistic and its arguments for a given type."""

        # 1. Type Compatibility
        if not AnalyticsRegistry.check_stat_compatibility(stat, element_type):
            if is_explicit:
                raise FedbiomedError(
                    f"Statistic '{stat}' is not valid for type {element_type.value}"
                )
            return False  # Skip invalid default

        # 2. Argument Validation
        try:
            AnalyticsRegistry.validate_args(stat, element_type, args)
        except FedbiomedError as e:
            raise FedbiomedError(
                f"Invalid arguments for statistic '{stat}': {e}"
            ) from e

        return True

    def _resolve_and_validate_roots(
        self,
        element_type: DatasetElementType,
        requested_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Resolves root statistics and validates argument consistency.

        Returns a flat config containing only root statistics (no expansion), ensuring
        that arguments between roots and their implied dependencies are consistent.

        Args:
            element_type: The type of dataset element (ROW or IMAGE).
            requested_config: Map of requested statistics to their arguments.

        Returns:
            Flat dict with root stats and their args.
        """
        # Map to track arguments for every stat (explicit or implied)
        # stat_name -> (stat_args, source_stat_name)
        stat_arg_map: Dict[str, Tuple[Dict[str, Any], str]] = {}

        for stat, stat_args in requested_config.items():
            # Check if this stat was already implied by a previous stat in the loop
            if stat in stat_arg_map:
                existing_args, source = stat_arg_map[stat]
                if existing_args != stat_args:
                    raise FedbiomedError(
                        f"Conflicting arguments for statistic '{stat}': "
                        f"implied by '{source}' with {existing_args}, "
                        f"but explicitly requested with {stat_args}"
                    )

            # Record it (overwrite is fine as we checked equality or it's new)
            stat_arg_map[stat] = (stat_args, stat)

            # Check dependencies
            # get_dependencies returns all recursive dependencies
            dependencies = AnalyticsRegistry.get_dependencies(stat, element_type)

            for dep in dependencies:
                # Validate that dependency accepts these arguments
                try:
                    AnalyticsRegistry.validate_args(dep, element_type, stat_args)
                except FedbiomedError as e:
                    raise FedbiomedError(
                        f"Statistic '{stat}' implies dependency '{dep}', but arguments are invalid for '{dep}': {e}"
                    ) from e

                if dep in stat_arg_map:
                    existing_args, source = stat_arg_map[dep]
                    if existing_args != stat_args:
                        raise FedbiomedError(
                            f"Conflicting arguments for dependency '{dep}': "
                            f"required by '{source}' with {existing_args}, "
                            f"but required by '{stat}' with {stat_args}"
                        )
                else:
                    stat_arg_map[dep] = (stat_args, stat)

        # Get roots
        # Roots are stats in specific requested list that are not implied by others in the list
        roots = AnalyticsRegistry.get_roots(requested_config.keys(), element_type)

        # Construct final config from roots
        final_config = {root: requested_config[root] for root in roots}

        return final_config
