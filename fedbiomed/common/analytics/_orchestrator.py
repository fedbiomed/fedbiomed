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
        fa_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Computes the requested statistics over the dataset.

        Args:
            dataset: The dataset to compute statistics for.
            subschema: Selection to filter the schema (e.g. subset of columns/keys).
            stats: Default list of statistics to compute (e.g. ['mean', 'std']).
            fa_args: Specific arguments for statistics, structured matching the schema.
                     e.g. {'image': {'histogram': {'bin_edges': [...]}}}

        Returns:
            The computed statistics.

        Raises:
            FedbiomedError: If validation fails or dataset capability is missing.
        """
        # 1. Check Capability
        if not all(
            hasattr(dataset, _) for _ in ("get_analytics_schema", "get_analytics_item")
        ):
            raise FedbiomedError(
                "Dataset does not implement 'get_analytics_schema' and/or 'get_analytics_item'."
            )

        # 2. Get Schema
        schema = dataset.get_analytics_schema()

        # 3. Build & Validate Configuration
        config = self._build_and_validate_config(schema, dataset_schema, stats, fa_args)

        # 4. Build Accumulator Tree
        accumulator = self._create_accumulator(config)

        # 5. Iterate and Accumulate
        n_samples = len(dataset)
        for idx in range(n_samples):
            sample = dataset.get_analytics_item(idx)
            accumulator.update(sample)

        # 6. Finalize
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
        if config["type"] in ("list", "tuple"):
            children = [
                self._create_accumulator(item) for item in config.get("children", [])
            ]
            return SequenceAccumulator(children, is_tuple=(config["type"] == "tuple"))

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
        args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validates inputs and builds the configuration tree in a single pass."""
        # Dispatch based on schema type
        if isinstance(schema, dict):
            return self._handle_dict(schema, subschema, stats, args)
        if isinstance(schema, (list, tuple)):
            return self._handle_sequence(schema, subschema, stats, args)

        # Leaf Handling
        element_type = schema.type if isinstance(schema, DatasetElementSpec) else None
        if element_type == DatasetElementType.ROW:
            return self._handle_row(schema, subschema, stats, args)
        if element_type == DatasetElementType.IMAGE:
            # Image does not support subschema selection
            if subschema is not None:
                raise FedbiomedError(
                    "Subschema selection is not applicable for IMAGE type."
                )
            return self._handle_image(stats, args)

        raise FedbiomedError(f"Unsupported schema type or element type: {type(schema)}")

    def _handle_dict(
        self,
        schema: Dict,
        subschema: Optional[Union[List[Union[str, Dict[str, Any]]], Dict[str, Any]]],
        stats: Optional[str],
        args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Validate Args
        if args is not None:
            if not isinstance(args, dict):
                raise FedbiomedError(
                    f"Args for dict schema must be a dict. Got {type(args)}."
                )
            invalid = set(args.keys()) - set(schema.keys())
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
            child_args = args.get(k) if args else None
            children[k] = self._build_and_validate_config(
                schema[k], child_sub, stats, child_args
            )

        return {"type": "dict", "children": children}

    def _handle_sequence(
        self,
        schema: Union[List, Tuple],
        subschema: Optional[Union[List, Tuple]],
        stats: Optional[List[str]],
        args: Optional[Union[List, Tuple]],
    ) -> Dict[str, Any]:
        # Validate Subschema
        if subschema is not None:
            if not isinstance(subschema, (list, tuple)):
                raise FedbiomedError(
                    f"Subschema for sequence must be list/tuple. Got {type(subschema)}."
                )
            if len(subschema) != len(schema):
                raise FedbiomedError(
                    "Subschema length mismatch. Pass None to ignore elements in list/tuple."
                )

        # Validate Args
        if args is not None:
            if not isinstance(args, (list, tuple)):
                raise FedbiomedError(
                    f"Args for sequence must be list/tuple. Got {type(args)}."
                )
            if len(args) != len(schema):
                raise FedbiomedError(
                    "Args length mismatch. Pass None to ignore elements in list/tuple."
                )

        children = []
        for idx, item_schema in enumerate(schema):
            child_sub = subschema[idx] if subschema else None
            child_args = args[idx] if args else None
            children.append(
                self._build_and_validate_config(
                    item_schema, child_sub, stats, child_args
                )
            )

        return {"type": "tuple", "children": children}

    def _handle_row(
        self,
        schema: RowSpec,
        subschema: Optional[List[str]],
        stats: Optional[List[str]],
        args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Ensure Schema Type
        if not isinstance(schema, RowSpec):
            raise FedbiomedError("Schema for ROW type must be a RowSpec instance.")

        # Validate Subschema
        if subschema is not None:
            if not isinstance(subschema, (list, tuple)):
                raise FedbiomedError("Subschema for ROW must be a list of columns.")
            invalid = set(subschema) - set(schema.columns)
            if invalid:
                raise FedbiomedError(f"Invalid columns in subschema: {invalid}")
        selected_cols = subschema if subschema is not None else schema.columns

        # Validate Args
        if args is not None:
            if not isinstance(args, dict):
                raise FedbiomedError("Args for ROW must be a dict.")
            invalid = set(args.keys()) - set(schema.columns)
            if invalid:
                raise FedbiomedError(f"Invalid columns in args: {invalid}")

        # Compile Config
        col_configs = {}
        for col in selected_cols:
            col_args = args.get(col) if args else None
            col_configs[col] = self._compile_leaf_stats(
                DatasetElementType.ROW, stats, col_args
            )

        return {
            "type": DatasetElementType.ROW,
            "conf": col_configs,
            "columns": selected_cols,  # Preserve column order for accumulators
        }

    def _handle_image(
        self, stats: Optional[List[str]], args: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validates and builds config for IMAGE type."""
        if args is not None and not isinstance(args, dict):
            raise FedbiomedError("Args for IMAGE must be a dict.")

        stats_config = self._compile_leaf_stats(DatasetElementType.IMAGE, stats, args)
        return {"type": DatasetElementType.IMAGE, "stats": stats_config}

    def _compile_leaf_stats(
        self,
        element_type: DatasetElementType,
        stats: Optional[List[str]],
        args: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compiles valid statistics configuration for a leaf node.

        Returns a flat config with requested root statistics.
        Primitives are implicitly handled by the Accumulator based on root stats.
        """
        requested_config = {}
        args = args or {}
        stats = stats or []

        candidates = set(stats).union(args.keys())

        # Validate and Filter Candidates
        for stat in candidates:
            # Check explicit presence or default
            is_explicit = stat in args
            current_args = args.get(stat, {})

            if self._validate_leaf_stat(element_type, stat, current_args, is_explicit):
                requested_config[stat] = current_args

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
        except ValueError as e:
            if is_explicit:
                raise FedbiomedError(
                    f"Invalid arguments for statistic '{stat}': {e}"
                ) from e
            return False  # Skip default with missing args

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
        # 1. Map to track arguments for every stat (explicit or implied)
        # stat_name -> (args, source_stat_name)
        stat_arg_map: Dict[str, Tuple[Dict[str, Any], str]] = {}

        for stat, args in requested_config.items():
            # Check if this stat was already implied by a previous stat in the loop
            if stat in stat_arg_map:
                existing_args, source = stat_arg_map[stat]
                if existing_args != args:
                    raise FedbiomedError(
                        f"Conflicting arguments for statistic '{stat}': "
                        f"implied by '{source}' with {existing_args}, "
                        f"but explicitly requested with {args}"
                    )

            # Record it (overwrite is fine as we checked equality or it's new)
            stat_arg_map[stat] = (args, stat)

            # Check dependencies
            # get_dependencies returns all recursive dependencies
            dependencies = AnalyticsRegistry.get_dependencies(stat, element_type)

            for dep in dependencies:
                # Validate that dependency accepts these arguments
                try:
                    AnalyticsRegistry.validate_args(dep, element_type, args)
                except FedbiomedError as e:
                    raise FedbiomedError(
                        f"Statistic '{stat}' implies dependency '{dep}', but arguments are invalid for '{dep}': {e}"
                    ) from e

                if dep in stat_arg_map:
                    existing_args, source = stat_arg_map[dep]
                    if existing_args != args:
                        raise FedbiomedError(
                            f"Conflicting arguments for dependency '{dep}': "
                            f"required by '{source}' with {existing_args}, "
                            f"but required by '{stat}' with {args}"
                        )
                else:
                    stat_arg_map[dep] = (args, stat)

        # 2. Get roots
        # Roots are stats in specific requested list that are not implied by others in the list
        roots = AnalyticsRegistry.get_roots(requested_config.keys(), element_type)

        # 3. Construct final config from roots
        final_config = {root: requested_config[root] for root in roots}

        return final_config
