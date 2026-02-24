# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
import uuid
from collections import OrderedDict
from typing import Any, Callable, Iterator, Optional

from fedbiomed.common.analytics import AGGREGATORS_MAP
from fedbiomed.common.constants import ErrorNumbers, Stats
from fedbiomed.common.dataset import validate_dataset_args
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import FAReply
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.jobs import FARequestJob
from fedbiomed.researcher.requests import Requests


class FAResult:
    """Stores per-node analytics results and computes global aggregations."""

    def __init__(self, replies: dict[str, FAReply]) -> None:
        """Initialise from a mapping of per-node FA replies.

        Args:
            replies: Mapping of node_id to FAReply.
        """
        self._data: dict[str, Any] = {}  # {node_id: raw_orchestrator_output}
        if replies:
            self._parse_replies(replies)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_reply(node_id: str, reply: FAReply) -> None:
        """Raise :exc:`FedbiomedError` if *reply* carries no usable output."""
        if not reply.output:
            raise FedbiomedError(f"Node '{node_id}' returned empty output.")

    @staticmethod
    def _is_stat_leaf(obj: dict) -> bool:
        """Return ``True`` if *obj* is a stat-leaf dict.

        A stat-leaf dict contains at least one key registered in
        :data:`AGGREGATORS_MAP`.
        """
        return any(k in AGGREGATORS_MAP for k in obj)

    @staticmethod
    def _traverse_stat_leaves(obj: Any) -> Iterator[dict]:
        """Yield every stat-leaf dict found in *obj* (depth-first).

        Structural dicts and sequences are traversed recursively. ``None``
        elements within sequences are skipped. Yields stop at stat-leaf dicts
        — their contents are not descended into further.

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
        """Return all keys found inside stat-leaf dicts (any depth, first node).

        Stat-leaf dicts are identified by having at least one key present in
        :data:`AGGREGATORS_MAP`.  All keys of those dicts are collected,
        including helper keys (e.g. ``"bin_edges"``) that are not themselves
        registered aggregators.

        Returns:
            Set of raw stat key strings present at leaf positions.
        """
        if not self._data:
            return set()
        return {
            k
            for leaf in FAResult._traverse_stat_leaves(self._first_output)
            for k in leaf
        }

    def _parse_replies(self, replies: dict[str, FAReply]) -> None:
        """Store each node's raw output verbatim."""
        for node_id, reply in replies.items():
            FAResult._validate_reply(node_id, reply)
            self._data[node_id] = reply.output

    @property
    def _first_output(self) -> Any:
        """Raw output of the first stored node. Only call when ``_data`` is non-empty."""
        return next(iter(self._data.values()))

    @staticmethod
    def _deep_merge(existing: Any, new: Any) -> Any:
        """Recursively merge *new* into *existing*.

        Dicts are merged key-by-key. Same-typed sequences are merged
        element-wise; both sequences must have the same length.

        Raises:
            ValueError: If *existing* and *new* are sequences of different
                lengths.
        """
        if isinstance(existing, dict) and isinstance(new, dict):
            result = dict(existing)
            for k, v in new.items():
                result[k] = FAResult._deep_merge(result[k], v) if k in result else v
            return result
        if isinstance(existing, (list, tuple)) and type(existing) is type(new):
            merged = [
                FAResult._deep_merge(e, n) for e, n in zip(existing, new, strict=True)
            ]
            return type(existing)(merged)
        # Scalar leaves are overwritten by the new value.
        return new

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def merge(self, replies: dict[str, FAReply]) -> None:
        """Add stats from a new request into the existing output tree,
        preserving all previously stored stats.

        Args:
            replies: Mapping of node_id to FAReply from the new request.
        """
        for node_id, reply in replies.items():
            FAResult._validate_reply(node_id, reply)
            if node_id not in self._data:
                self._data[node_id] = reply.output
            else:
                self._data[node_id] = FAResult._deep_merge(
                    self._data[node_id], reply.output
                )

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
            A nested structure mirroring the data schema, where each
            stat-leaf position is represented as ``{}``.  Returns ``None``
            when no data has been stored yet.
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
            return {}

        return _schema(self._first_output)

    def has_stat(self, stat_name: str) -> bool:
        """Return ``True`` if *stat_name* is present at every stat-leaf of every node's output.

        ``None`` elements within sequence outputs are skipped: a sequence position
        holding ``None`` does not cause the stat to be considered absent.
        """
        if not self._data:
            return False
        for out in self._data.values():
            leaves = list(FAResult._traverse_stat_leaves(out))
            if not leaves or not all(stat_name in leaf for leaf in leaves):
                return False
        return True

    def available_stats(self) -> list[str]:
        """Return statistic names found anywhere in the first node's output.

        Only names that have a registered aggregator in :data:`AGGREGATORS_MAP`
        are included, so internal helper keys are excluded.

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
        This includes both stats that are directly stored (e.g. ``"mean"``) and
        derived stats (e.g. ``"std"`` when ``mean``, ``variance``, and ``count``
        are all present, even though ``std`` is not itself a stored key).

        Returns:
            Sorted list of computable stat names.
        """
        raw_keys = self._leaf_stat_keys()
        result = [
            stat_name
            for stat_name, fn in AGGREGATORS_MAP.items()
            if set(inspect.signature(fn).parameters).issubset(raw_keys)
        ]
        return sorted(result)

    # ------------------------------------------------------------------
    # Access & Aggregation
    # ------------------------------------------------------------------

    def node_stats(self, node_id: str) -> Any:
        """Return the raw statistics output for *node_id*.

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
        return self._data[node_id]

    def all_node_stats(
        self,
        include_node_id: bool = True,
    ) -> dict[str, Any] | list[Any]:
        """Return statistics for every node in a single call.

        Args:
            include_node_id: When ``True`` (default) the result is a dict
                keyed by node ID.  When ``False`` a plain list of per-node
                outputs is returned in the same order as :attr:`node_ids`.

        Returns:
            When *include_node_id* is ``True``: ``{node_id: output}``.
            When *include_node_id* is ``False``: ``[output, ...]``.
        """
        if include_node_id:
            return dict(self._data)
        return list(self._data.values())

    @staticmethod
    def _aggregate_tree(
        outputs: list[Any],
        aggregator: Callable[..., Any],
    ) -> Any:
        """Recursively aggregate *outputs* using *aggregator* at every stat-leaf dict.

        Dicts containing at least one registered stat key are treated as stat-leaf
        nodes and passed directly to *aggregator*. Structural dicts and sequences
        are traversed recursively.

        Args:
            outputs: List of per-node output values, all sharing the same structure.
            aggregator: Callable registered in :data:`AGGREGATORS_MAP` for the
                target statistic.

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

        Given a mapping from stat name to its aggregated tree (all trees share
        the same structure), produces a single tree where every leaf position
        becomes a ``{stat_name: value}`` dict for every stat.

        Args:
            stat_results: Mapping ``{stat_name: aggregated_tree}`` where all
                trees share identical structure.

        Returns:
            A single tree whose leaves are ``{stat_name: value}`` dicts.
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

        The result preserves the schema structure of the node output: every
        stat-leaf dict is replaced by the value returned by the registered
        aggregator.  Structural dicts and sequences are traversed recursively.

        Stat-leaf dicts are detected by the presence of any key that appears
        in :data:`AGGREGATORS_MAP`, so *derived* stats (e.g. ``"std"`` when
        the data contains ``mean``, ``variance``, and ``count``) are supported
        even when the stat's name is not itself a stored key.

        Args:
            stat_name: Name of the statistic to aggregate (e.g. ``"mean"``).
                When ``None`` (default), all computable statistics are
                aggregated and the result preserves the original data-tree
                structure with stat names grouped at the leaves.  When
                provided, the name must appear in :meth:`computable_stats`.

        Returns:
            When *stat_name* is ``None``: the aggregated tree where every
            stat-leaf position becomes ``{stat_name: value}`` for every
            computable statistic, e.g.
            ``{"price": {"mean": 42.5, "count": 100}}``.
            Otherwise: aggregated value with the same shape as the node output
            tree.

        Raises:
            FedbiomedError: If *stat_name* is not computable, or if a node
                output contains an unexpected scalar leaf.
        """
        computable = self.computable_stats()
        all_outputs = list(self._data.values())

        if stat_name is None:
            per_stat = {
                s: FAResult._aggregate_tree(all_outputs, AGGREGATORS_MAP[s])
                for s in computable
            }
            return FAResult._merge_stat_results(per_stat)

        if stat_name not in computable:
            raise FedbiomedError(
                f"Statistic '{stat_name}' is not computable from the stored results. "
                f"Computable stats: {computable}."
            )

        return FAResult._aggregate_tree(all_outputs, AGGREGATORS_MAP[stat_name])


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
        # Session-level dataset arguments applied to every call. Set via set_dataset_args()
        self._dataset_args: Optional[dict] = None
        # Cache: maps a hash to a FAResult that accumulates stats with the same argument combination
        self._results_store: OrderedDict[str, FAResult] = OrderedDict()

    @property
    def fa_id(self) -> str:
        """Unique ID of this federated analytics instance."""
        return self._fa_id

    @property
    def dataset_args(self) -> Optional[dict]:
        """Session-level dataset arguments, or ``None`` if none are set."""
        return self._dataset_args

    def set_dataset_args(self, dataset_args: Optional[dict]) -> None:
        """Set session-level dataset arguments.

        Args:
            dataset_args: Keyword arguments forwarded to the dataset on every analytics
                call.  Pass ``None`` to clear the session default.
        """
        if dataset_args is not None:
            dataset_type = next(iter(self._fds.data().values())).get("data_type")
            validate_dataset_args(dataset_type, dataset_args)
        self._dataset_args = dataset_args

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
        dataset_args: Optional[dict],
        dataset_schema: Optional[str | list[str | dict]],
        fa_args: Optional[dict],
    ) -> str:
        """Create a stable string key from the node list, dataset and FA arguments.

        Args:
            node_ids: Current list of participating node IDs.
            dataset_args: Dataset filtering/selection arguments.
            dataset_schema: Schema definition.
            fa_args: FA-specific computation arguments.

        Returns:
            A hex digest string that uniquely identifies the argument combination.
        """
        key_data = {
            "node_ids": sorted(node_ids),
            "dataset_args": dataset_args or {},
            "dataset_schema": dataset_schema,
            "fa_args": fa_args or {},
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True, default=str).encode(),
            usedforsecurity=False,
        ).hexdigest()

    def compute_analytics(
        self,
        stats: str | list[str],
        dataset_args: Optional[dict] = None,
        dataset_schema: Optional[str | list[str | dict]] = None,
        fa_args: Optional[dict] = None,
    ) -> FAResult:
        """Compute analytics across nodes. Statistics that are already cached are not re-requested.

        Args:
            stats: Statistic name(s) to compute.
            dataset_args: If provided, updates the session-level dataset arguments via
                :meth:`set_dataset_args` before executing the request.
            dataset_schema: Schema definition.
            fa_args: Federated analytics arguments.

        Returns:
            A :class:`FAResult` containing per-node data and supporting global aggregation.
        """
        if isinstance(stats, str):
            stats = [stats]

        node_ids = self.get_node_ids()
        if dataset_args is not None:
            self.set_dataset_args(dataset_args)
        cache_key = FederatedAnalytics.make_cache_key(
            node_ids, self.dataset_args, dataset_schema, fa_args
        )
        cached = self._results_store.get(cache_key)

        missing = [s for s in stats if cached is None or not cached.has_stat(s)]

        if missing:
            fa_job = FARequestJob(
                fa_id=self._fa_id,
                fa_args=fa_args,
                stats=missing,
                dataset_args=self.dataset_args,
                dataset_schema=dataset_schema,
                federated_dataset=self._fds,
                experiment_id=self._experiment_id,
                researcher_id=self._researcher_id,
                requests=self._reqs,
                nodes=node_ids,
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
                    f"{len(errors)} node(s) returned errors ({list(errors)}). "
                    "No results available."
                )

            if cached is None:
                cached = FAResult(analytics_replies)
            else:
                cached.merge(analytics_replies)

            self._results_store[cache_key] = cached
            if len(self._results_store) > self._MAX_CACHE_SIZE:
                self._results_store.popitem(last=False)  # evict oldest (FIFO)

        return cached

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def mean(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Compute the mean across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.MEAN.value,
            dataset_schema=dataset_schema,
        )

    def variance(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Compute the variance across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.VARIANCE.value,
            dataset_schema=dataset_schema,
        )

    def min(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Compute the minimum across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.MIN.value,
            dataset_schema=dataset_schema,
        )

    def max(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Compute the maximum across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.MAX.value,
            dataset_schema=dataset_schema,
        )

    def count(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
    ) -> FAResult:
        """Compute the count of valid samples across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.COUNT.value,
            dataset_schema=dataset_schema,
        )

    def histogram(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
        fa_args: Optional[dict] = None,
    ) -> FAResult:
        """Compute histograms across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.
            fa_args: FA-specific arguments (e.g. ``bin_edges``).

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.HISTOGRAM.value,
            dataset_schema=dataset_schema,
            fa_args=fa_args,
        )

    def quantile(
        self,
        dataset_schema: Optional[str | list[str | dict]] = None,
        fa_args: Optional[dict] = None,
    ) -> FAResult:
        """Compute quantiles across all nodes.

        Args:
            dataset_schema: Schema used to interpret the dataset.
            fa_args: FA-specific arguments (e.g. ``quantiles``).

        Returns:
            A :class:`FAResult`.
        """
        return self.compute_analytics(
            stats=Stats.QUANTILE.value,
            dataset_schema=dataset_schema,
            fa_args=fa_args,
        )
