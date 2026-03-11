# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Module defining the FederatedAnalytics class, which manages federated analytics workflows within an Experiment."""

import copy
import hashlib
import inspect
import json
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from fedbiomed.common.analytics import AGGREGATORS_MAP
from fedbiomed.common.constants import ErrorNumbers, SecureAggregationSchemes, Stats
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import FAReply
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.jobs import FARequestJob
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.secagg import SecureAggregation


class FAResult:
    """Stores per-node analytics results and computes global aggregations."""

    def __init__(self, replies: Optional[dict[str, FAReply]]) -> None:
        """Initialise from a mapping of per-node FA replies.

        Args:
            replies: Mapping of node_id to FAReply.
        """
        self._data: dict[str, Any] = {}  # {node_id: raw_orchestrator_output}
        # Cache for the list of computable stats, set to None on every merge.
        self._computable_stats_cache: Optional[list[str]] = None
        if replies:
            self.merge(replies)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stat_leaf(obj: dict) -> bool:
        """Return 'True' if all dict keys are registered in 'AGGREGATORS_MAP'."""
        return isinstance(obj, dict) and obj and all(k in AGGREGATORS_MAP for k in obj)

    @staticmethod
    def _traverse_stat_leaves(obj: Any) -> Iterator[dict]:
        """Yield every stat-leaf dict found in *obj* (depth-first).

        Args:
            obj: A nested structure of dicts, lists, tuples, and scalar leaves.
        """
        # pyStructural dicts and sequences are traversed recursively
        if isinstance(obj, dict):
            if FAResult._is_stat_leaf(obj):
                yield obj
            else:
                for v in obj.values():
                    yield from FAResult._traverse_stat_leaves(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if item is not None:
                    yield from FAResult._traverse_stat_leaves(item)

    def _leaf_stat_keys(self) -> set[str]:
        """Return Set of raw stat key strings present at leaf positions (any depth, first node)."""
        if not self._data:
            return set()
        return {
            k
            for leaf in FAResult._traverse_stat_leaves(self._first_output)
            for k in leaf
        }

    @property
    def _first_output(self) -> Any:
        """Raw output of the first stored node.

        Raises:
            FedbiomedError: If no data has been stored yet.
        """
        if not self._data:
            raise FedbiomedError("FAResult contains no node data.")
        return next(iter(self._data.values()))

    @staticmethod
    def _deep_merge(existing: Any, new: Any) -> Any:
        """Recursively merge *new* into *existing*.

        Raises:
            FedbiomedError: If *existing* and *new* are sequences of different
                lengths, or if their types are structurally incompatible.
        """
        # pyStructural dicts and sequences are are merged element-wise
        if isinstance(existing, dict) and isinstance(new, dict):
            result = dict(existing)
            for k, v in new.items():
                result[k] = FAResult._deep_merge(result[k], v) if k in result else v
            return result
        if isinstance(existing, (list, tuple)) and type(existing) is type(new):
            try:
                merged = [
                    FAResult._deep_merge(e, n)
                    for e, n in zip(existing, new, strict=True)
                ]
            except ValueError as e:
                raise FedbiomedError(
                    f"Cannot merge sequences of different lengths "
                    f"({len(existing)} vs {len(new)})."
                ) from e
            return type(existing)(merged)
        # Raise instead of silently overwriting when types are structurally incompatible.
        if isinstance(existing, (dict, list, tuple)) or isinstance(
            new, (dict, list, tuple)
        ):
            raise FedbiomedError(
                f"Cannot deep-merge incompatible types: "
                f"{type(existing).__name__} and {type(new).__name__}."
            )
        # Both scalars: new overwrites.
        return new

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def merge(self, replies: dict[str, FAReply]) -> None:
        """Add stats from a new request into the existing output tree,
        preserving all previously stored stats.

        Args:
            replies: Mapping of node_id to FAReply from the new request.

        Raises:
            FedbiomedError: If any reply carries no usable output, or if the new output
                is structurally incompatible with the existing data.
        """
        # Validate everything before touching _data (avoid partial-state).
        for node_id, reply in replies.items():
            if reply.output is None:
                raise FedbiomedError(f"Node '{node_id}' returned None output.")

        for node_id, reply in replies.items():
            if node_id not in self._data:
                self._data[node_id] = reply.output
            else:
                self._data[node_id] = FAResult._deep_merge(
                    self._data[node_id], reply.output
                )
        self._computable_stats_cache = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def node_ids(self) -> list[str]:
        """List of node IDs present in the result."""
        return list(self._data.keys())

    @property
    def schema(self) -> Any:
        """Schema of the result tree, useful for inspecting output structure without raw values.

        Returns:
            A nested structure mirroring the data schema, where each stat-leaf position is represented as ``{}``.
        """
        if not self._data:
            return None

        def _schema(obj: Any) -> Any:
            if isinstance(obj, dict):
                if FAResult._is_stat_leaf(obj):
                    return {}
                return {k: _schema(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                items = [_schema(item) for item in obj]
                return type(obj)(items)
            return None

        return _schema(self._first_output)

    def has_stat(self, stat_name: str) -> bool:
        """Return ``True`` if *stat_name* is present at every stat-leaf of every node's output."""
        if not self._data:
            return False
        for out in self._data.values():
            leaves = list(FAResult._traverse_stat_leaves(out))
            if not leaves or not all(stat_name in leaf for leaf in leaves):
                return False
        return True

    def available_stats(self) -> list[str]:
        """Return statistic names found anywhere in the first node's output.

        Returns:
            Sorted list of statistic names.
        """
        if not self._data:
            return []
        return sorted(
            {
                k
                for leaf in FAResult._traverse_stat_leaves(self._first_output)
                for k in leaf
                if k in AGGREGATORS_MAP
            }
        )

    def computable_stats(self) -> list[str]:
        """Return names of aggregators that can be computed from the available data.

        A statistic is *computable* if every parameter required by its registered
        aggregator function is present as a key in the data's stat-leaf dicts.
        The result is cached and invalidated automatically when 'merge' is called.

        Returns:
            Sorted list of computable stat names.
        """
        if self._computable_stats_cache is not None:
            return self._computable_stats_cache
        raw_keys = self._leaf_stat_keys()
        result = sorted(
            stat_name
            for stat_name, fn in AGGREGATORS_MAP.items()
            if set(inspect.signature(fn).parameters).issubset(raw_keys)
        )
        self._computable_stats_cache = result
        return result

    # ------------------------------------------------------------------
    # Access & Aggregation
    # ------------------------------------------------------------------

    def node_stats(self, node_id: str) -> Any:
        """Return a copy of the raw statistics output for *node_id*.

        Args:
            node_id: Identifier of the node.

        Returns:
            Per-node output mirroring the dataset's schema
            (e.g. ``{col: {stat: val}}`` for row data, ``{stat: val}`` for image data).

        Raises:
            FedbiomedError: If *node_id* is not found.
        """
        if node_id not in self._data:
            raise FedbiomedError(
                f"Node '{node_id}' not found in results. "
                f"Available nodes: {self.node_ids}."
            )
        return copy.deepcopy(self._data[node_id])

    def all_node_stats(self) -> dict[str, Any]:
        """Return a Dict mapping each node ID to a deep copy of its output."""
        return {
            node_id: copy.deepcopy(output) for node_id, output in self._data.items()
        }

    @staticmethod
    def _aggregate_tree(
        outputs: list[Any],
        aggregator: Callable[..., Any],
    ) -> Any:
        """Recursively aggregate *outputs* using *aggregator* at every stat-leaf dict.

        Args:
            outputs: List of per-node output values, all sharing the same structure.
            aggregator: Callable registered in 'AGGREGATORS_MAP' for the target statistic.

        Raises:
            FedbiomedError: If a node output contains an unexpected bare scalar leaf.
        """
        first = outputs[0]
        if isinstance(first, dict):
            if FAResult._is_stat_leaf(first):
                all_keys = {k for o in outputs for k in o}
                kwargs = {s: [o[s] for o in outputs if s in o] for s in all_keys}
                return aggregator(**kwargs)
            return {
                k: FAResult._aggregate_tree([o[k] for o in outputs], aggregator)
                for k in first
            }
        if isinstance(first, (list, tuple)):
            result = [
                FAResult._aggregate_tree([o[i] for o in outputs], aggregator)
                for i in range(len(first))
            ]
            return type(first)(result)
        raise FedbiomedError(
            f"Unexpected scalar leaf encountered during aggregation: {first!r}. "
            "Node outputs should always be dicts or sequences."
        )

    @staticmethod
    def _merge_stat_results(stat_results: dict[str, Any]) -> Any:
        """Transpose ``{stat_name: tree}`` into tree-first grouping with stat names at leaves.

        Args:
            stat_results: Mapping ``{stat_name: aggregated_tree}`` where all
                trees share identical structure.

        Returns:
            A single tree whose leaves are ``{stat_name: value}`` dicts.

        Example:
            Input: {
                "mean": {"age": 47.2, "income": 55000},
                "count": {"age": 100, "income": 100}
            }
            Output: {
                "age": {"mean": 47.2, "count": 100},
                "income": {"mean": 55000, "count": 100}
            }
        """
        if not stat_results:
            return {}
        first = next(iter(stat_results.values()))
        if isinstance(first, dict):
            return {
                k: FAResult._merge_stat_results(
                    {s: tree[k] for s, tree in stat_results.items()}
                )
                for k in first
            }
        if isinstance(first, (list, tuple)):
            result = [
                FAResult._merge_stat_results(
                    {s: tree[i] for s, tree in stat_results.items()}
                )
                for i in range(len(first))
            ]
            return type(first)(result)
        # Scalar leaf: collect all stat values at this position
        return stat_results

    def global_stat(self, stat_name: str) -> Any:
        """Compute the globally aggregated value for *stat_name* across all nodes.

        The result preserves the schema structure of the node output.

        Args:
            stat_name: Name of the statistic to aggregate (e.g. ``"mean"``).
                Must appear in 'computable_stats'.

        Returns:
            Aggregated value with the same structural shape as the per-node output tree

        Raises:
            FedbiomedError: If no node data is stored, if *stat_name* is not
                computable, or if a node output contains an unexpected scalar leaf.
        """
        if not self._data:
            raise FedbiomedError(
                "Cannot compute global stat: FAResult contains no node data."
            )
        computable = self.computable_stats()
        if stat_name not in computable:
            raise FedbiomedError(
                f"Statistic '{stat_name}' is not computable from the stored results. "
                f"Computable stats: {computable}."
            )
        return FAResult._aggregate_tree(
            list(self._data.values()), AGGREGATORS_MAP[stat_name]
        )

    def global_stats(self) -> Any:
        """Compute globally aggregated values for all computable statistics.

        Aggregates every stat returned and merges the results into a single tree.

        Returns:
            A tree whose leaves are ``{stat_name: value}``

        Raises:
            FedbiomedError: If a node output contains an unexpected scalar leaf.
        """
        if not self._data:
            return {}
        all_outputs = list(self._data.values())
        per_stat = {
            s: FAResult._aggregate_tree(all_outputs, AGGREGATORS_MAP[s])
            for s in self.computable_stats()
        }
        return FAResult._merge_stat_results(per_stat)


class FederatedAnalytics:
    """
    A class to manage Federated Analytics (FA) workflows within an Experiment.
    FA workflows allow researchers to perform analytics tasks across federated datasets.

    Results are cached. When a statistic has already been fetched with identical arguments,
    it is served from the cache without contacting the nodes again.
    """

    _MAX_CACHE_SIZE: int = 32  # Default maximum number of cached result sets

    def __init__(
        self,
        fds: FederatedDataset,
        experiment_id: str,
        researcher_id: str,
        reqs: Requests,
        experimentation_folder: str,
        secagg: Union[bool, SecureAggregation, None] = None,
        **kwargs,
    ) -> None:
        """Initialise a federated analytics session.

        Args:
            fds: The federated dataset to run analytics on.
            experiment_id: Identifier of the parent experiment.
            researcher_id: Identifier of the researcher.
            reqs: Request handler used to communicate with nodes.
            experimentation_folder: Local folder for storing experiment artefacts.
            secagg: Secure aggregation configuration. If True, uses default LOM scheme.
                If False or None, no secure aggregation is used.
            **kwargs: Additional keyword arguments (reserved for future use).
        """
        self._fa_id: str = "FA_" + str(uuid.uuid4())
        self._fds = fds
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._reqs = reqs
        self._experimentation_folder = experimentation_folder
        self._secagg = self._init_secagg(secagg)
        # Cache: maps a hash to a FAResult that accumulates stats with the same argument combination
        self._results_store: OrderedDict[str, FAResult] = OrderedDict()

    def _init_secagg(
        self, secagg: Union[bool, SecureAggregation, None]
    ) -> Union[SecureAggregation, bool]:
        """Initialize secure aggregation.

        Args:
            secagg: Secure aggregation configuration.

        Returns:
            SecureAggregation instance if enabled, False otherwise.
        """
        if secagg is None or secagg is False:
            return False
        try:
            if isinstance(secagg, SecureAggregation):
                return secagg
        except TypeError:
            pass
        if isinstance(secagg, bool) and secagg:
            return SecureAggregation(
                scheme=SecureAggregationSchemes.LOM, active=True
            )
        return False

    @property
    def secagg(self) -> Union[SecureAggregation, bool]:
        """Return the secure aggregation instance."""
        return self._secagg

    def set_secagg(
        self,
        secagg: Union[bool, SecureAggregation],
        scheme: SecureAggregationSchemes = SecureAggregationSchemes.LOM,
    ) -> SecureAggregation:
        """Configure secure aggregation for federated analytics.

        Args:
            secagg: Whether to enable secure aggregation (True/False) or provide
                a pre-configured SecureAggregation instance.
            scheme: The secure aggregation scheme to use if secagg is True.

        Returns:
            The configured SecureAggregation instance.
        """
        if isinstance(secagg, bool):
            self._secagg = SecureAggregation(scheme=scheme, active=secagg)
        else:
            try:
                is_secagg = isinstance(secagg, SecureAggregation)
            except TypeError:
                is_secagg = False
            if is_secagg:
                self._secagg = secagg
            else:
                raise FedbiomedError(
                f"{ErrorNumbers.FB410.value}: Expected `secagg` argument bool or "
                f"`SecureAggregation` but got {type(secagg)}"
            )
        return self._secagg

    def secagg_setup(self, parties: list[str]) -> dict:
        """Setup secure aggregation context with participating nodes.

        Args:
            parties: List of party IDs (node IDs + researcher ID) participating
                in the secure aggregation.

        Returns:
            Dictionary containing secagg arguments to be passed to nodes.
        """
        if isinstance(self._secagg, bool):
            return {}
        if not self._secagg.active:
            return {}

        logger.info("Setting up secure aggregation for federated analytics...")
        if not self._secagg.setup(parties=parties):
            raise FedbiomedError(
                f"{ErrorNumbers.FB415.value}: Failed to setup secure aggregation "
                "for federated analytics."
            )

        secagg_arguments = self._secagg.train_arguments()
        return secagg_arguments

    @property
    def fa_id(self) -> str:
        """Unique ID of this federated analytics instance."""
        return self._fa_id

    def get_node_ids(self) -> list[str]:
        """Return the list of node IDs participating in this federated analytics.

        Returns:
            A list of node IDs
        """
        if self._fds is None:
            raise FedbiomedExperimentError(
                "No defined FederatedDataset found for FederatedAnalytics."
            )

        node_ids = self._fds.node_ids()
        if not node_ids:
            raise FedbiomedExperimentError(
                "Empty list of nodes for analytics: no nodes replied to original "
                "`federated_analytics_request` or sampling strategy returned an empty list."
            )

        return node_ids

    @staticmethod
    def make_cache_key(
        node_ids: list[str],
        dataset_schema: Optional[str | list[str | dict]],
        stats_args: Optional[dict],
    ) -> str:
        """Create a stable string key from the node list and FA arguments.

        Args:
            node_ids: Current list of participating node IDs.
            dataset_schema: Schema definition.
            stats_args: FA-specific computation arguments.

        Returns:
            A hex digest string that uniquely identifies the argument combination.
        """
        key_data = {
            "node_ids": sorted(node_ids),
            "dataset_schema": dataset_schema,
            "stats_args": stats_args or {},
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True, default=str).encode(),
            usedforsecurity=False,
        ).hexdigest()

    def fetch_stats(
        self,
        stats: Optional[str | list[str]] = None,
        dataset_schema: Optional[str | list[str | dict]] = None,
        stats_args: Optional[dict] = None,
    ) -> FAResult:
        """Fetch analytics across nodes. Statistics that are already cached are not re-requested.

        Args:
            stats: Statistic name(s) to request from nodes. Optional when ``stats_args``
                provides all necessary computation arguments.
            dataset_schema: Schema definition.
            stats_args: Federated analytics arguments.

        Returns:
            A :class:`FAResult` containing per-node data and supporting global aggregation.

        Raises:
            FedbiomedError: If both ``stats`` and ``stats_args`` are empty/None.
        """
        if not stats and not stats_args:
            raise FedbiomedError(
                "At least one of 'stats' or 'stats_args' must be provided."
            )

        if isinstance(stats, str):
            stats = [stats]

        node_ids = self.get_node_ids()
        cache_key = FederatedAnalytics.make_cache_key(
            node_ids, dataset_schema, stats_args
        )
        cached = self._results_store.get(cache_key)

        # When stats=None the request is entirely args-driven: only skip if already cached.
        # When stats is a list, only request the individual stats not yet in the cache.
        if stats is None:
            need_request = cached is None
            missing = None
        else:
            missing = [s for s in stats if cached is None or not cached.has_stat(s)]
            need_request = bool(missing)

        if need_request:
            secagg_active = (
                self._secagg is not False and self._secagg.active
            )

            secagg_arguments = {}
            if secagg_active:
                parties = [self._researcher_id] + node_ids
                secagg_arguments = self.secagg_setup(parties)
                # Pass the exact number of encrypting nodes (excluding researcher)
                # so nodes use the same n_users in JLS protect() as the researcher
                # sees in JLS aggregate() via len(list_y_u_tau).
                secagg_arguments["num_nodes"] = len(node_ids)

            fa_job = FARequestJob(
                fa_id=self._fa_id,
                stats_args=stats_args,
                stats=missing,
                dataset_schema=dataset_schema,
                federated_dataset=self._fds,
                experiment_id=self._experiment_id,
                researcher_id=self._researcher_id,
                requests=self._reqs,
                nodes=node_ids,
                secagg=secagg_active,
                secagg_arguments=secagg_arguments,
            )

            analytics_replies, errors = fa_job.execute()

            # Errors are logged but not raised
            for node_id, error in errors.items():
                logger.error(
                    "Error message received during analytics request for node "
                    f"{node_id} - {error.errnum}: {error.extra_msg}"
                )

            if not analytics_replies and errors:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: Federated analytics failed: all "
                    f"{len(errors)} node(s) returned errors ({str(errors)}). "
                    "No results available."
                )

            if secagg_active and analytics_replies:
                analytics_replies = self._decrypt_replies(analytics_replies)

            if cached is None:
                cached = FAResult(analytics_replies)
            else:
                cached.merge(analytics_replies)

            self._results_store[cache_key] = cached
            if len(self._results_store) > self._MAX_CACHE_SIZE:
                self._results_store.popitem(last=False)  # evict oldest (FIFO)

        return cached

    def _decrypt_replies(
        self, replies: dict[str, FAReply]
    ) -> dict[str, FAReply]:
        """Decrypt encrypted federated analytics replies.

        With SecAgg, the encrypted values from all nodes are aggregated first,
        then decrypted to produce the global result. Individual node contributions
        remain hidden.

        Returns a single-entry dict so that FAResult does not re-aggregate an
        already-global value (which would multiply it by the number of nodes).

        Args:
            replies: Dictionary of node_id -> FAReply with potentially encrypted output.

        Returns:
            Single-entry dictionary with the globally aggregated decrypted result.
        """
        if self._secagg is False:
            return replies

        secagg_params = self._get_secagg_params()
        if not secagg_params:
            logger.warning("SecAgg enabled but no parameters found, returning raw output")
            return replies

        try:
            global_output = self._aggregate_and_decrypt_all(replies, secagg_params)
        except Exception as e:
            logger.error(f"Failed to decrypt FA replies: {e}")
            raise FedbiomedError(
                f"{ErrorNumbers.FB415.value}: Failed to decrypt FA replies: {e}"
            )

        # Store the global result under a single node entry only.
        # Returning all nodes with the same output would cause FAResult.global_stat()
        # to aggregate the already-global value N times.
        first_node_id = next(iter(replies))
        replies[first_node_id].output = global_output
        return {first_node_id: replies[first_node_id]}

    def _aggregate_and_decrypt_all(
        self, replies: dict[str, FAReply], secagg_params: Dict
    ) -> Dict:
        """Aggregate encrypted values from all nodes and decrypt the result.

        This implements the core SecAgg logic: sum encrypted values from all nodes,
        then decrypt once to get the global result.

        Args:
            replies: Dictionary of node_id -> FAReply with encrypted output.
            secagg_params: Dictionary with key, biprime, clipping_range.

        Returns:
            Dictionary with globally aggregated (decrypted) values.
        """
        if not replies:
            return {}

        first_output = next(iter(replies.values())).output

        num_nodes = len(replies)

        # Stat keys whose global value is the SUM across nodes (not the average).
        # All other numeric stats are treated as averages (e.g. mean, variance).
        ADDITIVE_STAT_KEYS = {"count"}

        def get_encrypted_values_for_stat(replies: dict, stat_path: str) -> List:
            """Collect encrypted values for a specific stat path from all nodes."""
            values = []
            for reply in replies.values():
                val = reply.output
                for k in stat_path.split("."):
                    if isinstance(val, dict):
                        val = val.get(k)
                if val is None:
                    continue
                if isinstance(val, dict) and "_encrypted" in val:
                    values.append([val["value"]])
                elif isinstance(val, (int, float)):
                    values.append([int(val)])
            return values

        def process_node_output(output: Any, path: str = "") -> Any:
            """Process output, replacing encrypted values with global aggregated result."""
            if isinstance(output, dict):
                result = {}
                for k, v in output.items():
                    current_path = f"{path}.{k}" if path else k
                    if k == "histogram":
                        result[k] = self._decrypt_histogram_global(
                            replies, v, secagg_params, num_nodes, path=current_path
                        )
                    elif isinstance(v, dict) and "_encrypted" in v:
                        enc_vals = get_encrypted_values_for_stat(replies, current_path)
                        if enc_vals:
                            is_additive = k in ADDITIVE_STAT_KEYS
                            decrypted = self._decrypt_single_value(
                                enc_vals, secagg_params, num_nodes, is_additive=is_additive
                            )
                            result[k] = decrypted
                        else:
                            result[k] = v.get("value", 0)
                    else:
                        result[k] = process_node_output(v, current_path)
                return result
            elif isinstance(output, list):
                return [process_node_output(item, path) for item in output]
            return output

        return process_node_output(first_output)

    def _decrypt_histogram_global(
        self,
        replies: dict[str, FAReply],
        histogram: Dict,
        secagg_params: Dict,
        num_nodes: int,
        path: str = "",
    ) -> Dict:
        """Decrypt histogram by aggregating encrypted counts from all nodes.

        Histogram counts are additive (global = sum of per-node counts), so after
        decrypting the average (total_sample_size=num_nodes) we multiply by num_nodes
        to recover the actual sum.

        Args:
            replies: Dictionary of node_id -> FAReply.
            histogram: Histogram structure (from first node) with bin_edges and counts.
            secagg_params: Dictionary with key, biprime, clipping_range.
            num_nodes: Number of participating nodes.
            path: Dot-separated key path from the output root to this histogram
                (e.g. ``"col.histogram"``), used to locate the histogram in each
                node reply regardless of nesting depth.

        Returns:
            Decrypted histogram with aggregated counts.
        """
        if not isinstance(histogram, dict):
            return histogram

        bin_edges = histogram.get("bin_edges", [])

        # Collect per-node encrypted count vectors.
        # all_encrypted_counts[i] = [enc_bin0, enc_bin1, ..., enc_binK] from node i.
        # This is exactly the shape aggregate() expects: params[i] is node i's vector.
        all_encrypted_counts = []
        has_encrypted = False

        path_keys = [k for k in path.split(".") if k]
        for reply in replies.values():
            # Navigate to the histogram dict using the stored path
            val = reply.output
            for k in path_keys:
                if isinstance(val, dict):
                    val = val.get(k)
            hist = val
            if isinstance(hist, dict) and "counts" in hist:
                counts = hist.get("counts", [])
                if counts and isinstance(counts[0], dict) and "_encrypted" in counts[0]:
                    has_encrypted = True
                    all_encrypted_counts.append([c.get("value", 0) for c in counts])

        if not has_encrypted or not all_encrypted_counts:
            return histogram

        try:
            from fedbiomed.common.secagg import SecaggCrypter

            crypter = SecaggCrypter()
            num_bins = len(all_encrypted_counts[0])

            # aggregate() divides by total_sample_size in the quantized domain before
            # reverse-quantizing, which correctly recovers the per-node average.
            # Multiply by num_nodes afterwards to obtain the actual sum of counts.
            decrypted = crypter.aggregate(
                current_round=1,
                num_nodes=num_nodes,
                params=all_encrypted_counts,
                key=secagg_params["key"],
                biprime=secagg_params["biprime"],
                total_sample_size=num_nodes,
                clipping_range=secagg_params.get("clipping_range"),
                num_expected_params=num_bins,
            )

            return {"bin_edges": bin_edges, "counts": [int(d * num_nodes) for d in decrypted]}
        except Exception as e:
            logger.warning(f"Failed to decrypt histogram: {e}")
            return histogram

    def _decrypt_single_value(
        self,
        encrypted_values: List[List[int]],
        secagg_params: Dict,
        num_nodes: int,
        is_additive: bool = False,
    ) -> float:
        """Decrypt a single aggregated value.

        aggregate() divides by total_sample_size in the quantized domain before
        reverse-quantizing, which correctly recovers the per-node average.
        For additive stats (e.g. count) we multiply the result by num_nodes to
        recover the actual sum across nodes.

        Args:
            encrypted_values: List of [encrypted_value] from each node.
            secagg_params: Dictionary with key, biprime, clipping_range.
            num_nodes: Number of nodes.
            is_additive: If True, multiply the decrypted average by num_nodes to get
                the global sum. If False, return the average as-is.

        Returns:
            Decrypted aggregated value (sum or average depending on is_additive).
        """
        try:
            from fedbiomed.common.secagg import SecaggCrypter

            crypter = SecaggCrypter()

            decrypted = crypter.aggregate(
                current_round=1,
                num_nodes=num_nodes,
                params=encrypted_values,
                key=secagg_params["key"],
                biprime=secagg_params["biprime"],
                total_sample_size=num_nodes,
                clipping_range=secagg_params.get("clipping_range"),
                num_expected_params=1,
            )

            value = decrypted[0] if decrypted else 0
            return value * num_nodes if is_additive else value
        except Exception as e:
            logger.warning(f"Failed to decrypt value: {e}")
            return 0

    def _get_secagg_params(self) -> Optional[Dict]:
        """Get secagg parameters from the secagg instance.

        Returns:
            Dictionary with key, biprime, and clipping_range.
        """
        if self._secagg is False:
            return None

        try:
            secagg_context = self._secagg._secagg
            if hasattr(secagg_context, "_biprime"):
                return {
                    "key": getattr(secagg_context, "_key", 0),
                    "biprime": secagg_context._biprime,
                    "clipping_range": getattr(secagg_context, "_secagg_clipping_range", None),
                }
        except Exception as e:
            logger.warning(f"Failed to get secagg params: {e}")

        return None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def mean(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Return the global mean across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.MEAN.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("mean")

    def variance(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Return the global variance across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.VARIANCE.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("variance")

    def min(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Return the global minimum across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.MIN.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("min")

    def max(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Return the global maximum across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.MAX.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("max")

    def count(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Return the global count across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.COUNT.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("count")
