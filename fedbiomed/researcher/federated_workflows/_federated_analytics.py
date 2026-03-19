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
from fedbiomed.common.constants import Stats
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.message import FAReply
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.jobs import FARequestJob
from fedbiomed.researcher.requests import Requests


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
        """Yield every stat-leaf dict found in *obj* (depth-first).

        Args:
            obj: A nested structure of dicts, lists, tuples, and scalar leaves.
        """
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
        """Return a dict mapping each node ID to a deep copy of its output."""
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
                Must appear in :meth:`computable_stats`.

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

        Returns:
            A tree whose leaves are ``{stat_name: value}`` dicts.

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
    """Manages Federated Analytics (FA) workflows across a federated dataset.

    Requests are dispatched to nodes via :class:`FARequestJob`. Results are cached
    by ``(node_ids, dataset_schema, stats_args)``; repeated calls with the same
    arguments are served from cache without contacting nodes.
    """

    _MAX_CACHE_SIZE: int = 32  # Maximum number of cached result sets.

    def __init__(
        self,
        fds: FederatedDataset,
        experiment_id: str,
        researcher_id: str,
        reqs: Requests,
        experimentation_folder: str,
        **kwargs,
    ) -> None:
        """Initialise a federated analytics session.

        Args:
            fds: The federated dataset to run analytics on.
            experiment_id: Identifier of the parent experiment.
            researcher_id: Identifier of the researcher.
            reqs: Request handler used to communicate with nodes.
            experimentation_folder: Local folder for storing experiment artefacts.
            **kwargs: Additional keyword arguments (reserved for future use).
        """
        self._fa_id: str = "FA_" + str(uuid.uuid4())
        self._fds = fds
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._reqs = reqs
        self._experimentation_folder = experimentation_folder
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
            "stats_args": stats_args,
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True, default=str).encode(),
            usedforsecurity=False,
        ).hexdigest()

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
        )
        analytics_replies, errors = fa_job.execute()
        # Node-level errors are already logged by FARequestJob.execute()
        if not analytics_replies or errors:
            raise FedbiomedError(
                "No successful replies received or at least one node returned an error."
            )
        if cached is None:
            cached = FAResult(analytics_replies)
        else:
            cached.merge(analytics_replies)
        self._results_store[cache_key] = cached
        if len(self._results_store) > self._MAX_CACHE_SIZE:
            self._results_store.popitem(last=False)  # evict oldest (FIFO)
        return cached

    def fetch_stats(
        self,
        stats: str | list[str],
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Fetch named statistics across nodes. Already-cached statistics are not re-requested.

        Args:
            stats: Statistic name(s) to request from nodes (e.g. ``"mean"`` or
                ``["mean", "variance"]``).
            dataset_schema: Optional schema definition for filtering the dataset.

        Returns:
            A :class:`FAResult` containing per-node data and supporting global aggregation.
        """
        if isinstance(stats, str):
            stats = [stats]
        if not stats:
            raise FedbiomedError("'stats' must be a non-empty string or list.")

        node_ids = self.get_node_ids()
        cache_key = FederatedAnalytics.make_cache_key(node_ids, dataset_schema, None)
        cached = self._results_store.get(cache_key)

        missing = [s for s in stats if cached is None or not cached.has_stat(s)]
        if missing:
            cached = self._execute_and_update_cache(
                cache_key, missing, None, dataset_schema, node_ids, cached
            )

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
                or ``{"col": {"quantile": {"q": 0.5}}}``\).

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

        return cached

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def count(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> Any:
        """Return the global count across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.COUNT.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("count")

    def mean(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> Any:
        """Return the global mean across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.MEAN.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("mean")

    def variance(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> Any:
        """Return the global variance across nodes."""
        fa_result = self.fetch_stats(
            stats=Stats.VARIANCE.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("variance")

    def std(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> Any:
        """Return the global standard deviation across nodes."""
        # std is derived from variance+mean+count; requesting variance primitives is sufficient.
        fa_result = self.fetch_stats(
            stats=Stats.VARIANCE.value,
            dataset_schema=dataset_schema,
        )
        return fa_result.global_stat("std")
