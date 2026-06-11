# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Federated Analytics workflow: FAResult (per-node results and aggregation) and FederatedAnalytics (request orchestration and caching)."""

import copy
import hashlib
import inspect
import json
import uuid
from collections import OrderedDict
from typing import Any, Callable, Iterator, Optional

from fedbiomed.common.analytics import AGGREGATORS_MAP
from fedbiomed.common.constants import SAParameters, Stats
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import FAReply
from fedbiomed.common.utils import unflatten_fa_output
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
        """Return ``True`` if all dict keys are registered in ``AGGREGATORS_MAP``."""
        return isinstance(obj, dict) and obj and all(k in AGGREGATORS_MAP for k in obj)

    @staticmethod
    def _traverse_stat_leaves(obj: Any) -> Iterator[dict]:
        """Yield every stat-leaf dict found in *obj* (depth-first)."""
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
        """Return the set of stat key names at leaf positions across the first node's output."""
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
    def _filter_output(output: Any, schema: list) -> Optional[Any]:
        """Return a filtered copy of *output* keeping only keys listed in *schema*.

        Returns ``None`` if *output* is not a filterable dict, a key is absent, a child
        schema is given for a non-dict value, or a schema item type is unsupported.
        """
        # Stat-leaves are terminal values, not filterable structural dicts.
        if not isinstance(output, dict) or FAResult._is_stat_leaf(output):
            return None
        result: dict = {}
        for item in schema:
            if isinstance(item, str):
                key, child_sub = item, None
            elif isinstance(item, dict) and len(item) == 1:
                key, child_sub = next(iter(item.items()))
            else:
                return None  # unsupported schema item type
            if key not in output:
                return None
            if isinstance(child_sub, str):
                child_sub = [child_sub]  # normalise bare-string child schema to list
            if child_sub is None:
                result[key] = copy.deepcopy(
                    output[key]
                )  # no sub-schema: copy subtree as-is
            elif not isinstance(output[key], dict):
                return None  # child schema given but value is not a filterable dict
            elif isinstance(child_sub, (list, tuple)):
                child = FAResult._filter_output(output[key], list(child_sub))
                if child is None:
                    return None
                result[key] = child
            else:
                return None  # child_sub is an unexpected type
        return result

    def _filtered_copy(self, schema: list) -> Optional["FAResult"]:
        """Return a copy filtered to *schema*, or ``None`` if any node output cannot be filtered."""
        filtered_data: dict[str, Any] = {}
        for node_id, output in self._data.items():
            out = FAResult._filter_output(output, schema)
            if out is None:
                return None
            filtered_data[node_id] = out
        result = FAResult(None)
        result._data = filtered_data
        return result

    @staticmethod
    def _deep_merge(existing: Any, new: Any) -> Any:
        """Recursively merge *new* into *existing*.

        Raises:
            FedbiomedError: If *existing* and *new* are sequences of different
                lengths, or if their types are structurally incompatible.
        """
        # Structural dicts and sequences are merged element-wise
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
        # Incompatible structural types cannot be merged.
        if isinstance(existing, (dict, list, tuple)) or isinstance(
            new, (dict, list, tuple)
        ):
            raise FedbiomedError(
                f"Cannot deep-merge incompatible types: "
                f"{type(existing).__name__} and {type(new).__name__}."
            )
        # Both scalars: new overwrites.
        return new

    @staticmethod
    def _combine(a: "FAResult", b: "FAResult") -> "FAResult":
        """Return a new FAResult deep-merging the per-node trees of *a* and *b*.

        Used to fold the round-1 result (count/sum/…) and the round-2 centered
        result (count/sum_sq_centered) into a single result from which
        count, mean, variance and std are all computable. Both results must share
        the same node keys (same nodes, schema and — under secagg — the
        ``__secagg__`` virtual node).
        """
        combined = FAResult(None)
        combined._data = {nid: copy.deepcopy(out) for nid, out in a._data.items()}
        for nid, out in b._data.items():
            if nid in combined._data:
                combined._data[nid] = FAResult._deep_merge(combined._data[nid], out)
            else:
                combined._data[nid] = copy.deepcopy(out)
        return combined

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _merge_aggregate(self, output: dict) -> None:
        """Merge a pre-built aggregate output into the secagg virtual node.

        Used by the encrypted reply path where all node contributions have been
        decrypted and unflattened into a single aggregate dict by the researcher.
        """
        if "__secagg__" not in self._data:
            self._data["__secagg__"] = output
        else:
            self._data["__secagg__"] = FAResult._deep_merge(
                self._data["__secagg__"], output
            )
        self._computable_stats_cache = None

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
                    return {}  # stat-leaf: replace values with empty sentinel
                return {k: _schema(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                items = [_schema(item) for item in obj]
                return type(obj)(items)
            return None

        return _schema(self._first_output)

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
        The result is cached and invalidated automatically when :meth:`merge` is called.

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

    def node_stats(self, node_id: Optional[str] = None) -> Any:
        """Return a copy of the raw statistics output for one or all nodes.

        Args:
            node_id: Identifier of the node. When omitted, returns a dict
                mapping every node ID to its output.

        Returns:
            Per-node output when *node_id* is given, or a ``{node_id: output}``
            dict for all nodes when omitted.

        Raises:
            FedbiomedError: If *node_id* is not found.
        """
        if node_id is None:
            return {nid: copy.deepcopy(output) for nid, output in self._data.items()}

        if node_id not in self._data:
            raise FedbiomedError(
                f"Node '{node_id}' not found in results. "
                f"Available nodes: {self.node_ids}."
            )
        return copy.deepcopy(self._data[node_id])

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

    def global_stats(self, stat_name: Optional[str] = None) -> Any:
        """Compute globally aggregated value(s) across all nodes.

        Args:
            stat_name: Statistic to aggregate (e.g. ``"mean"``). Must appear in
                :meth:`computable_stats`. When omitted, all computable statistics
                are aggregated and merged into a ``{stat_name: value}`` tree.

        Returns:
            Aggregated value for the given stat, or a merged tree of all computable
            statistics when *stat_name* is omitted (``{}`` if no data is stored).

        Raises:
            FedbiomedError: If no node data is stored, *stat_name* is unknown or
                not computable from available data.
        """
        if stat_name is None:
            if not self._data:
                return {}
            all_outputs = list(self._data.values())
            per_stat = {
                s: FAResult._aggregate_tree(all_outputs, AGGREGATORS_MAP[s])
                for s in self.computable_stats()
            }
            return FAResult._merge_stat_results(per_stat)

        if not self._data:
            raise FedbiomedError(
                "Cannot compute global stat: FAResult contains no node data."
            )
        computable = self.computable_stats()
        if stat_name not in computable:
            if stat_name not in AGGREGATORS_MAP:
                raise FedbiomedError(
                    f"Statistic '{stat_name}' is not a valid statistic. "
                    f"Valid statistics: {sorted(AGGREGATORS_MAP)}."
                )
            available_keys = self._leaf_stat_keys()
            required = set(inspect.signature(AGGREGATORS_MAP[stat_name]).parameters)
            missing = sorted(required - available_keys)
            raise FedbiomedError(
                f"Statistic '{stat_name}' cannot be computed: missing required data {missing}. "
                f"Available data keys: {sorted(available_keys)}. "
                f"Fetch the missing data with: FederatedAnalytics.fetch_stats({missing})"
            )
        return FAResult._aggregate_tree(
            list(self._data.values()), AGGREGATORS_MAP[stat_name]
        )


class FederatedAnalytics:
    """Manages Federated Analytics (FA) workflows across a federated dataset.

    Requests are dispatched to nodes via :class:`FARequestJob`. Results are cached
    by ``(node_ids, dataset_schema, stats_args)``; repeated calls with the same
    arguments are served from cache without contacting nodes.
    """

    _MAX_CACHE_SIZE: int = 32  # Maximum number of cached result sets.

    # Statistics that are not computable by nodes in a single round
    _CENTERED_DERIVED: tuple[str, ...] = ("variance", "std")

    def __init__(
        self,
        fds: FederatedDataset,
        experiment_id: str,
        researcher_id: str,
        reqs: Requests,
        experimentation_folder: str,
        secagg: Optional[SecureAggregation] = None,
        **kwargs,
    ) -> None:
        """Initialise a federated analytics session.

        Args:
            fds: The federated dataset to run analytics on.
            experiment_id: Identifier of the parent experiment.
            researcher_id: Identifier of the researcher.
            reqs: Request handler used to communicate with nodes.
            experimentation_folder: Local folder for storing experiment artefacts.
            secagg: Optional ``SecureAggregation`` instance.  When ``None`` or
                inactive, FA runs in plaintext.
            **kwargs: Additional keyword arguments (reserved for future use).
        """
        self._fa_id: str = "FA_" + str(uuid.uuid4())
        self._fds = fds
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._reqs = reqs
        self._experimentation_folder = experimentation_folder
        self._secagg: SecureAggregation = (
            secagg if secagg is not None else SecureAggregation(active=False)
        )
        self._fa_round_counter: int = 0
        # Maps hash(node_ids, dataset_schema, stats_args) → FAResult.
        self._results_store: OrderedDict[str, FAResult] = OrderedDict()

    @property
    def fa_id(self) -> str:
        """Unique ID of this federated analytics instance."""
        return self._fa_id

    def get_node_ids(self) -> list[str]:
        """Return the node IDs participating in this federated analytics session."""
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
    def _sort_schema(obj):
        """Recursively sort dicts by key and lists by JSON repr, so list order is irrelevant."""
        if isinstance(obj, dict):
            return {
                k: FederatedAnalytics._sort_schema(v) for k, v in sorted(obj.items())
            }
        elif isinstance(obj, list):

            def _norm(item):
                if isinstance(item, dict) and len(item) == 1:
                    k, v = next(iter(item.items()))
                    return {
                        k: FederatedAnalytics._sort_schema(
                            [v] if isinstance(v, str) else v
                        )
                    }  # normalise "x" → ["x"]
                return FederatedAnalytics._sort_schema(item)

            return sorted(
                (_norm(i) for i in obj),
                key=lambda x: json.dumps(x, sort_keys=True, default=str),
            )
        else:
            return obj

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
            "dataset_schema": FederatedAnalytics._sort_schema(dataset_schema),
            "stats_args": stats_args,
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True, default=str).encode(),
            usedforsecurity=False,
        ).hexdigest()

    def _cache_store(self, key: str, result: FAResult) -> None:
        """Insert *result* at *key* and evict the oldest entry if over capacity."""
        self._results_store[key] = result
        if len(self._results_store) > self._MAX_CACHE_SIZE:
            self._results_store.popitem(last=False)

    def _secagg_setup(self, node_ids: list[str]) -> dict:
        """Set up secure aggregation for the given nodes and return secagg arguments.

        Args:
            node_ids: Node IDs that will participate in the FA round.

        Returns:
            Secagg arguments dict (suitable for inclusion in FARequest), or an empty
            dict when secure aggregation is not active.
        """
        if not self._secagg.active:
            return {}
        self._secagg.setup(
            parties=node_ids,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            insecure_validation=False,
        )
        # Send both fixed FA ranges so the nodes can validate them against their own.
        self._secagg.clipping_range = SAParameters.FA_CLIPPING_RANGE
        secagg_arguments = dict(self._secagg.train_arguments())
        secagg_arguments["secagg_target_range"] = SAParameters.FA_TARGET_RANGE
        return secagg_arguments

    def _execute_and_update_cache(
        self,
        cache_key: str,
        stats: Optional[list[str]],
        stats_args: Optional[dict],
        dataset_schema: Optional[str | list[str | dict]],
        node_ids: list[str],
        cached: Optional[FAResult],
    ) -> FAResult:
        """Dispatch a :class:`FARequestJob` to *node_ids* and write the result into the cache.

        Args:
            cache_key: Key under which the result is stored in ``_results_store``.
            stats: Named statistics to request, or ``None`` when using *stats_args*.
            stats_args: Structured analytics arguments, or ``None`` when using *stats*.
            dataset_schema: Optional schema filter forwarded to the job.
            node_ids: Nodes that will receive the request.
            cached: Existing :class:`FAResult` to merge new replies into, or ``None``
                to create a fresh result from the replies.

        Returns:
            The updated (or newly created) :class:`FAResult`, stored at *cache_key*.

        Raises:
            FedbiomedError: If no replies are received or if any node returns an error.
        """
        secagg_arguments = self._secagg_setup(node_ids)
        self._fa_round_counter += 1
        if secagg_arguments:
            secagg_arguments["fa_round"] = self._fa_round_counter

        fa_job = FARequestJob(
            fa_id=self._fa_id,
            stats=stats,
            stats_args=stats_args,
            dataset_schema=dataset_schema,
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
            secagg_arguments=secagg_arguments or None,
        )

        # Node-level errors and empty replies are handled by FARequestJob.execute()
        analytics_replies = fa_job.execute()

        first_reply = next(iter(analytics_replies.values()))
        if first_reply.encrypted:
            model_params = {
                nid: r.params_encrypted for nid, r in analytics_replies.items()
            }
            num_expected_params = len(first_reply.params_encrypted)
            num_nodes = len(analytics_replies)
            aggregated_flat = self._secagg.aggregate(
                round_=self._fa_round_counter,
                total_sample_size=num_nodes,
                model_params=model_params,
                num_expected_params=num_expected_params,
                target_range=SAParameters.FA_TARGET_RANGE,
            )
            # Crypter returns cross-node mean (÷num_nodes cancels quantization
            # offset); ×num_nodes restores the additive sum FA aggregators expect.
            aggregated_flat = [v * num_nodes for v in aggregated_flat]
            output_schema = first_reply.output_schema
            aggregated_output = unflatten_fa_output(aggregated_flat, output_schema)
            if cached is None:
                cached = FAResult(None)
            cached._merge_aggregate(aggregated_output)
        else:
            if cached is None:
                cached = FAResult(analytics_replies)
            else:
                cached.merge(analytics_replies)

        self._cache_store(cache_key, cached)
        return cached

    def _recover_from_cache(
        self,
        cache_key: Any,
        missing: list[str],
        dataset_schema: list[str | dict],
    ) -> Optional[FAResult]:
        """Recover missing stats from a cached superset-schema result, or None.

        Scans stored results for an entry containing all ``missing`` stats whose
        schema is a superset of ``dataset_schema``. ``_filtered_copy`` returns None
        on any missing key, so a non-None result implies a superset. On success the
        filtered copy is stored under ``cache_key`` and returned.
        """
        for other in self._results_store.values():
            satisfiable = set(other.available_stats()) | set(other.computable_stats())
            if not all(s in satisfiable for s in missing):
                continue
            filtered = other._filtered_copy(dataset_schema)
            if filtered is not None:
                self._cache_store(cache_key, filtered)
                logger.info(
                    f"Statistics {missing} recovered from cached result "
                    "— skipping node requests."
                )
                return filtered
        return None

    @staticmethod
    def _centered_args_from_mean(mean_tree: Any) -> Any:
        """Turn a global-mean tree into ``sum_sq_centered`` stats_args."""
        if isinstance(mean_tree, dict):
            return {
                k: FederatedAnalytics._centered_args_from_mean(v)
                for k, v in mean_tree.items()
            }
        if isinstance(mean_tree, (list, tuple)):
            return type(mean_tree)(
                FederatedAnalytics._centered_args_from_mean(v) for v in mean_tree
            )
        # Scalar leaf: the global mean for one feature.
        return {"sum_sq_centered": {"mean": float(mean_tree)}}

    def _fetch_centered(
        self,
        stats: list[str],
        dataset_schema: Optional[list[str | dict]],
        node_ids: list[str],
    ) -> FAResult:
        """Run the two-pass centered scheme and return a combined result."""
        # Round 1: the requested stats plus the count+sum the global mean is built from
        round1_stats = sorted(
            {s for s in stats if s not in self._CENTERED_DERIVED} | {"count", "sum"}
        )
        r1 = self.fetch_stats(round1_stats, dataset_schema, _emit_log=False)
        mean_tree = r1.global_stats("mean")

        # Round 2: Σ(x − μ)² centered on the round-1 global mean.
        # Reuses round 1's count, so round 2 only needs the centered moment.
        centered_args = self._centered_args_from_mean(mean_tree)
        round2_key = self.make_cache_key(node_ids, None, centered_args)
        r2 = self._results_store.get(round2_key)
        if r2 is None:
            r2 = self._execute_and_update_cache(
                round2_key, None, centered_args, None, node_ids, None
            )

        return FAResult._combine(r1, r2)

    def _log_global_stats(self, result: FAResult) -> None:
        """Log the aggregated global statistics of *result* once, prettily."""
        logger.info(
            "Global statistics:\n%s",
            json.dumps(result.global_stats(), indent=2, default=str, sort_keys=True),
        )

    def fetch_stats(
        self,
        stats: Optional[str | list[str]] = None,
        dataset_schema: Optional[str | list[str | dict]] = None,
        _emit_log: bool = True,
    ) -> FAResult:
        """Fetch named statistics across nodes. Already-cached statistics are not re-requested.

        Args:
            stats: Statistic name(s) to request from nodes (e.g. ``"mean"``).
                Defaults to ``["count", "mean", "variance"]``.
            dataset_schema: Optional schema definition for filtering the dataset.
            _emit_log: Internal. When ``False``, suppresses the final global-stats
                log line — used for intermediate rounds (e.g. the round-1 mean of
                the variance/std two-pass) so the stats are logged once, at the end.

        Returns:
            A :class:`FAResult` containing per-node data and supporting global aggregation.
        """
        if stats is None:
            stats = ["count", "mean", "variance"]
        if isinstance(stats, str):
            stats = [stats]
        if not stats:  # guard against explicit empty list
            raise FedbiomedError("'stats' must be a non-empty string or list.")

        # Stats nodes can compute directly; the centered-derived stats
        # (variance/std) are handled here via the two-pass scheme. Any other
        # AGGREGATORS_MAP-only stat remains non-requestable.
        requestable = {s.value for s in Stats}
        computed_only = [
            s
            for s in stats
            if s in AGGREGATORS_MAP
            and s not in requestable
            and s not in self._CENTERED_DERIVED
        ]
        unknown = [
            s for s in stats if s not in AGGREGATORS_MAP and s not in requestable
        ]

        errors = []
        if unknown:
            errors.append(f"The following are not valid statistics: {unknown}.")
        if computed_only:
            all_prereqs = sorted(
                {
                    p
                    for s in computed_only
                    for p in inspect.signature(AGGREGATORS_MAP[s]).parameters
                    if p in requestable
                }
            )
            errors.append(
                f"The following statistics are derived and cannot be requested from nodes directly: {computed_only}. "
                f"To compute them, call fetch_stats({all_prereqs}) first, then FAResult.global_stats()."
            )
        if errors:
            raise FedbiomedError(" ".join(errors))

        if isinstance(dataset_schema, str):  # str → [str]
            dataset_schema = [dataset_schema]

        node_ids = self.get_node_ids()
        cache_key = FederatedAnalytics.make_cache_key(node_ids, dataset_schema, None)
        cached = self._results_store.get(cache_key)

        # Two-pass path: any requested variance/std needs the global mean first.
        derived = [s for s in stats if s in self._CENTERED_DERIVED]
        if derived:
            if cached is not None and set(stats).issubset(
                set(cached.computable_stats())
            ):
                logger.info(
                    f"All requested statistics {stats} are already cached — "
                    "skipping node requests."
                )
                result = cached
            else:
                result = self._fetch_centered(stats, dataset_schema, node_ids)
                self._cache_store(cache_key, result)
            if _emit_log:
                self._log_global_stats(result)
            return result

        # Re-requesting the same (node_ids, schema) key returns identical data, so a
        # stat only needs requesting if it is present in NO form in the cache.
        satisfiable = (
            set(cached.available_stats()) | set(cached.computable_stats())
            if cached is not None
            else set()
        )
        missing = [s for s in stats if s not in satisfiable]
        if missing:
            recovered = (
                self._recover_from_cache(cache_key, missing, dataset_schema)
                if dataset_schema is not None and cached is None
                else None
            )
            if recovered is not None:
                cached = recovered
            else:
                cached = self._execute_and_update_cache(
                    cache_key, missing, None, dataset_schema, node_ids, cached
                )
        else:
            logger.info(
                f"All requested statistics {stats} are already cached — "
                "skipping node requests."
            )

        if _emit_log:
            self._log_global_stats(cached)

        return cached

    def fetch_stats_with_args(
        self,
        stats_args: dict,
    ) -> FAResult:
        """Fetch analytics driven entirely by structured arguments across nodes.

        Unlike :meth:`fetch_stats`, this method does not accept named statistics or a
        separate ``dataset_schema`` — the schema selection and computation parameters
        are both encoded within *stats_args*.

        Results are cached by ``(node_ids, stats_args)``. The same args always produce
        the same result; different args always trigger a new request.

        Args:
            stats_args: Structured analytics arguments that encode both schema selection
                and computation parameters (e.g.
                ``{"image": {"histogram": {"bin_edges": [0, 128, 256]}}}``
                or ``{"col": {"quantile": {"q": 0.5}}}``).

        Returns:
            A :class:`FAResult` containing per-node data and supporting global aggregation.

        Raises:
            FedbiomedError: If ``stats_args`` is empty or None, or if all nodes return errors.
        """
        if not stats_args:
            raise FedbiomedError("'stats_args' must be provided and non-empty.")

        node_ids = self.get_node_ids()
        cache_key = FederatedAnalytics.make_cache_key(node_ids, None, stats_args)
        cached = self._results_store.get(cache_key)

        if cached is None:
            cached = self._execute_and_update_cache(
                cache_key, None, stats_args, None, node_ids, None
            )
        else:
            logger.info(
                "Exact stats_args request already cached — skipping node requests."
            )

        return cached

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def _stat(
        self,
        fetch_stat: str,
        aggregate_stat: str,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> Any:
        """Fetch *fetch_stat* primitives then aggregate *aggregate_stat* from the result."""
        return self.fetch_stats(fetch_stat, dataset_schema).global_stats(aggregate_stat)

    def count(self, dataset_schema: Optional[str | list[str | dict]] = None) -> Any:
        """Return the global count across nodes."""
        return self._stat(Stats.COUNT.value, "count", dataset_schema)

    def mean(self, dataset_schema: Optional[str | list[str | dict]] = None) -> Any:
        """Return the global mean across nodes."""
        return self._stat(Stats.MEAN.value, "mean", dataset_schema)

    def variance(self, dataset_schema: Optional[str | list[str | dict]] = None) -> Any:
        """Return the global variance across nodes (two-pass centered scheme)."""
        return self.fetch_stats("variance", dataset_schema).global_stats("variance")

    def std(self, dataset_schema: Optional[str | list[str | dict]] = None) -> Any:
        """Return the global standard deviation across nodes (two-pass centered scheme)."""
        return self.fetch_stats("std", dataset_schema).global_stats("std")
