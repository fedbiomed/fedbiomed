# Federated Analytics

This page describes the architecture of the Federated Analytics (FA) feature, the responsibility of each component, and which files to touch when making changes.

---

## How It Works

```
Researcher                         Node
──────────                         ────
FederatedAnalytics.fetch_stats()
  └─ FARequestJob.execute()  ──── FARequest ──►  node.py
                                                   └─ FAJob.run()
                                                        ├─ dataset.compute_stats()
                                                        │    └─ AnalyticsOrchestrator
                                                        │         └─ Accumulators
                                                        └─ FAReply 
  └─ FAResult (aggregate)  ◄────────────────────────────────┘
```

1. The researcher calls a stat method (e.g. `exp.analytics.mean()`).
2. `FederatedAnalytics` checks its cache; missing stats are sent as an `FARequest` to each node via `FARequestJob`.
3. On the node, `FAJob` validates the request and calls `dataset.compute_stats()`.
4. `AnalyticsOrchestrator` reads `analytics_schema()` from the dataset, builds an accumulator tree, iterates samples, and returns partial statistics.  
5. The node sends back an `FAReply`; the researcher side merges node replies into an `FAResult` and aggregates globally via `AGGREGATORS_MAP`.

---

## Statistics, API and Aggregation

### Available Statistics

`Stats` (in `fedbiomed/common/constants.py`) is the single source of truth for stat names. Every node accumulator, registry entry, and aggregator function must reference one of these values. Adding a stat starts here.

| Enum | String value | Required `stats_args` key | Notes |
|------|-------------|--------------------------|-------|
| `Stats.COUNT` | `"count"` | — | Per-column non-null count (ROW), or shape count (IMAGE) |
| `Stats.MEAN` | `"mean"` | — | auto-requests `count` as dependency |
| `Stats.VARIANCE` | `"variance"` | — | auto-requests `mean` + `count` |
| `Stats.HISTOGRAM` | `"histogram"` | `bin_edges` | - |

!!! note "Researcher-only derived stats"
    `std` and `sum` are computable on the researcher side from `mean`/`variance`/`count` via `FAResult.global_stat()` — they are never sent from nodes.

`FederatedAnalytics` is the entry point for all analytics requests. It handles caching and delegates network I/O to `FARequestJob`. `fetch_stats` is its core method — it resolves which stats are missing from the cache, sends an `FARequest` to each node for those stats only, merges the replies into an `FAResult`, and returns it.

```python
fetch_stats(
    stats: str | list[str] | None = None,     # one or more Stats string values
    dataset_schema: str | list | None = None, # column/modality filter; None → whole schema
    stats_args: dict | None = None,           # required for histogram
) -> FAResult
```

!!! note "See also"
    For the shape and accepted values of `dataset_schema`, see [Federated Analytics — Datasets](../user-guide/datasets/federated-analytics.md#making-a-custom-dataset-fa-compatible).

Convenience methods are thin wrappers around `fetch_stats` + `global_stat` for the most common stats:

```python
exp.analytics.mean(dataset_schema=None)
# → {'year': 2016.96, 'price': 16235.20, 'mileage': 23908.94, 'mpg': 55.25, ...}

exp.analytics.mean(dataset_schema=["price", "mileage"])
# → {'price': 16235.20, 'mileage': 23908.94}
```

!!! note "No `stats_args` in convenience methods"
    Use `fetch_stats` directly for `histogram`, which require `bin_edges` in `stats_args`.

### Stat Dependencies

The orchestrator resolves dependencies automatically before building the accumulator tree. This means requesting `variance` will also compute `mean` and `count` on the node even if they are not listed explicitly in `stats`. Dependencies and required arguments for each stat are declared in `fedbiomed/common/analytics/accumulators/_registry.py`.

### Cross-Node Aggregation

Once all nodes have replied, `FAResult` calls `AGGREGATORS_MAP` to combine per-node partial results into a single global value per modality or column. Each function is registered via the `@aggregator(stat)` decorator; its parameter names match `Stats` string values. 

| Stat | Aggregation logic |
|------|------------------|
| `count` | sum (scalar int, or dict of per-key counts) |
| `sum` | Σ(mean × count) per node |
| `mean` | weighted mean: Σ(mean × count) / Σcount |
| `variance` | combined sample variance via SS-within + SS-between |
| `std` | √variance (derived; never sent from nodes) |
| `histogram` | element-wise count sum (bin edges must match across nodes) |
| `quantile` | linear interpolation on the aggregated histogram |

---

## Component Responsibilities

### Common layer

| File | Responsibility |
|------|----------------|
| `fedbiomed/common/constants.py` | `Stats` enum — the single source of truth for valid stat names |
| `fedbiomed/common/message.py` | `FARequest` / `FAReply` wire schemas (add fields here when the protocol changes) |
| `fedbiomed/common/analytics/_aggregators.py` | `AGGREGATORS_MAP` — maps each stat name to its cross-node aggregation function |
| `fedbiomed/common/analytics/_orchestrator.py` | `AnalyticsOrchestrator` — drives per-node stat computation; builds accumulator trees from the dataset schema |
| `fedbiomed/common/analytics/accumulators/_registry.py` | Links stat names ↔ accumulator classes and element types; update here to register a new stat |
| `fedbiomed/common/analytics/accumulators/_operations.py` | Primitive accumulator implementations (sum, count, min, max, histogram, quantile, …) |
| `fedbiomed/common/analytics/accumulators/_row.py` | Vectorised accumulator for tabular / row data |
| `fedbiomed/common/analytics/accumulators/_image.py` | Accumulator for image data |
| `fedbiomed/common/analytics/accumulators/_base.py` | `Accumulator` abstract base class |

### Node layer

| File | Responsibility |
|------|----------------|
| `fedbiomed/node/config.py` | `allow_federated_analytics` flag — guards FA on a per-node basis |
| `fedbiomed/node/node.py` | Routes incoming `FARequest` messages to `FAJob` |
| `fedbiomed/node/jobs/_fa_job.py` | `FAJob` — validates the request, calls `dataset.compute_stats()`, returns `FAReply` or `ErrorMessage` |

### Researcher layer

| File | Responsibility |
|------|----------------|
| `fedbiomed/researcher/federated_workflows/_federated_analytics.py` | `FederatedAnalytics` (API, cache) and `FAResult` (per-node storage + cross-node aggregation) |
| `fedbiomed/researcher/federated_workflows/_federated_workflow.py` | Instantiates `FederatedAnalytics` as `experiment.analytics` |
| `fedbiomed/researcher/federated_workflows/jobs/_fa_request_job.py` | `FARequestJob` — broadcast `FARequest` to nodes and collect `FAReply` responses |

---

## Adding a New Statistic

1. **Add the name** to `Stats` in `fedbiomed/common/constants.py`.
2. **Implement the accumulator** in `fedbiomed/common/analytics/accumulators/_operations.py`.
3. **Register it** in `fedbiomed/common/analytics/accumulators/_registry.py` (element type, dependency, required args).
4. **Add the cross-node aggregator** to `AGGREGATORS_MAP` in `fedbiomed/common/analytics/_aggregators.py`.
5. *(Optional)* Add a convenience method in `FederatedAnalytics` (`fedbiomed/researcher/federated_workflows/_federated_analytics.py`).
6. **Add tests** in `tests/test_analytics/`.

---

## Test Coverage

| Test file | What it covers |
|-----------|----------------|
| `tests/test_analytics/test_federated_analytics.py` | `FederatedAnalytics` and `FAResult` (cache, merge, aggregation) |
| `tests/test_analytics/test_node_fa_job.py` | `FAJob` — permission checks, error paths, `compute_stats` delegation |
| `tests/test_analytics/test_fa_request_job.py` | `FARequestJob` — request broadcast and reply collection |
| `tests/test_analytics/test_analytics_orchestrator.py` | `AnalyticsOrchestrator` — schema parsing, accumulator wiring |
| `tests/test_analytics/test_aggregators.py` | `AGGREGATORS_MAP` aggregator functions |
| `tests/test_analytics/test_accumulators_row.py` | `RowAccumulator` |
| `tests/test_analytics/test_accumulators_image.py` | `ImageAccumulator` |
| `tests/test_analytics/test_accumulators_operations.py` | Primitive accumulator operations |
| `tests/test_analytics/test_accumulators_registry.py` | `AnalyticsRegistry` |
| `tests/test_message.py` | `FARequest` / `FAReply` message creation |
| `tests/test_node.py` | Node routing of `FARequest` to `FAJob` |

---

## Related Documentation

- User guide: [Federated Analytics](../user-guide/datasets/federated-analytics.md)
- Notebook tutorial: `notebooks/federated_analytics.ipynb`
- API reference: [Common Analytics](api/common/analytics.md) · [Node Jobs](api/node/jobs.md) · [Researcher Federated Workflows](api/researcher/federated_workflows.md)
