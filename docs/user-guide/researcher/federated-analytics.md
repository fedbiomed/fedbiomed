# Federated Analytics — Researcher

## Overview

**Federated Analytics (FA)** lets you compute statistics — such as means, variances, or histograms — across datasets held by multiple remote nodes, without the raw data ever leaving those nodes.

This page covers the **researcher side**: how to issue analytics requests and interpret the results.

> - For which datasets support FA and how to make a custom dataset FA-compatible, see [Federated Analytics — Datasets](../datasets/federated-analytics.md).
> - For enabling FA on a node, see [Federated Analytics — Nodes](../nodes/federated-analytics.md).

---

## How it works

### Request / compute / aggregate cycle

Every analytics call follows the same three-phase cycle:

1. **Request** — you send a statistics request to all participating nodes.
2. **Local compute** — each node runs the computation on its own data and sends back only the summary (not the raw data).
3. **Aggregate** — FedBioMed combines the per-node summaries into a single **global result** on the researcher side.

Because each node only ever returns aggregated summaries, the raw data never leaves the node.

### The dataset schema

Every FA-compatible dataset declares a **schema** — a description of what its samples look like. The schema is what the dataset's schema returns: column names for tabular data, or nested structures for multi-modal datasets (see [Federated Analytics — Datasets](../datasets/federated-analytics.md) for details on how datasets declare their schema).

When you call `fetch_stats()`, the analytics engine reads this schema from each node's dataset to know how to interpret each sample. You don't normally need to read or manipulate it directly — `dataset_schema` lets you select a subset of it (see below).

### The `Experiment` object and `analytics`

`Experiment` is the single entry point for both federated training and federated analytics (see [Experiment](experiment.md) for more). When you instantiate it with a set of tags, it contacts all connected nodes, discovers which ones hold a matching dataset, and collects the responses. As part of that setup it also creates a `FederatedAnalytics` object and exposes it as `exp.analytics`.

!!! note "FederatedAnalytics access"
    You never construct `FederatedAnalytics` directly; it is always accessed through the `analytics` attribute of an experiment. All network communication, result caching, and aggregation are handled internally — from your perspective, you call a method and get back a result.

### Caching

To avoid redundant network requests, results are cached per `(node_ids, dataset_schema, stats_args)` combination. When you repeat a call with the same parameters, the cached result is returned immediately — no messages are sent to the nodes.

---

## Hands on

### Convenience Methods

A **convenience method** is a thin wrapper around `fetch_stats` that covers the most common use-cases without requiring you to name the statistic explicitly. Each one sends the request, waits for all nodes to respond, aggregates the partial results, and returns the global value in a single call:

```python
count    = exp.analytics.count()     # total number of samples across all nodes
mean     = exp.analytics.mean()      # per-column (or per-pixel) mean
variance = exp.analytics.variance()  # per-column (or per-pixel) variance
```

The return value is a dict keyed by column name (for tabular data) or by modality name (for multi-modal data). For example, on a two-node tabular dataset with columns `year`, `price`, `mileage`, `mpg`, …:

```python
>>> exp.analytics.mean()
{'year': 2016.96, 'price': 16235.20, 'transmission': 1.02,
 'mileage': 23908.94, 'tax': 118.06, 'mpg': 55.25, 'engineSize': 1.57}

>>> exp.analytics.variance()
{'year': 4.40, 'price': 91583707.11, 'transmission': 0.31,
 'mileage': 444227213.03, 'tax': 4131.06, 'mpg': 138.72, 'engineSize': 0.33}

>>> exp.analytics.count()
{'year': 28632, 'price': 28633, 'transmission': 28633,
 'mileage': 28633, 'tax': 28633, 'mpg': 28633, 'engineSize': 28633}
```

!!! note "Convenience method limitations"
    - `count` can differ per column when a column has missing values.
    - `histogram` does not have a convenience method as it requires bin-edge arguments — use [`fetch_stats` with `stats_args`] (see below).

### Filtering with `dataset_schema`

By default, statistics are computed over everything the schema describes. Pass `dataset_schema` to restrict computation to a subset of columns or modalities.

**Tabular dataset — select columns by name:**

```python
# Only compute the mean over 'age' and 'bmi' columns
mean = exp.analytics.mean(dataset_schema=["age", "bmi"])
```

!!! note "Column names"
    The list must contain column names that exist in the dataset's schema.

**Multi-modal dataset — select modalities, and optionally columns within them:**

For datasets that expose multiple modalities (e.g. images paired with a clinical table), `dataset_schema` is either a list of modality names to include in full, or a dict mapping each modality name to a column selector for that modality (`None` means include all columns for that modality):

```python
# Include the whole 'T1' image modality and only 'age'/'bmi' from 'demographics'
mean = exp.analytics.mean(
    dataset_schema={"T1": None, "demographics": ["age", "bmi"]}
)
```

!!! note "Modality filtering"
    Any modality not mentioned in `dataset_schema` is excluded from the computation.


### Using `fetch_stats`

The convenience methods internally call `fetch_stats`. You can call it directly when you need full control over what is requested or want to inspect per-node results. It returns an `FAResult` object:

```python
result = exp.analytics.fetch_stats(
    stats="variance",              # statistic to compute — 'mean', 'variance', 'count'
    dataset_schema=["age", "bmi"], # optional column filter
)

# Globally aggregated value for a single named statistic
variance_result = result.global_stat("variance")

# All available stats merged into one nested dict: {column: {stat_name: value}}
all_stats = result.global_stats()

# Raw output from a single node — useful for debugging or per-site analysis
node_output = result.node_stats("node-id-1")
```

---

### Statistics Requiring Arguments

`histogram` needs to know the bin edges for each column upfront. The bins must be **identical across all nodes** so that the per-node counts can be summed into a global histogram. Supply them through `stats_args`:

```python
result = exp.analytics.fetch_stats(
    stats_args={
        "histogram": {
            "age":    {"bin_edges": [0, 20, 40, 60, 80, 100]},
            "income": {"bin_edges": [0, 25000, 50000, 75000, 100000]},
        }
    }
)
```

!!! note "`stats_args` and `stats`"
    `stats_args` and `stats` can be combined in a single call. At least one of the two must be provided.

---

## FAResult Reference

`fetch_stats` returns an `FAResult` object. Rather than returning a plain dict immediately, FedBioMed gives you this object for two reasons:

- **Lazy aggregation.** Per-node outputs are stored as-is. Global aggregation only happens when you call a method like `global_stat()`, so you pay the cost only when you need the result.
- **Incremental merging.** If you make a second `fetch_stats` call for a different statistic (but the same nodes and schema), the new results are merged into the same `FAResult` — no data is lost and no nodes are re-contacted for stats already in the cache.
- **Per-node access.** You can inspect what each individual node returned before aggregation, which is useful for detecting data quality issues or comparing site-specific distributions.

### What it stores

Internally, `FAResult` holds a dict mapping each node ID to that node's raw output tree. For a tabular dataset the output tree looks like:

```python
{
    "NODE_abc": {
        "age":    {"mean": 47.2, "count": 100, "variance": 120.5},
        "income": {"mean": 55000, "count": 100, "variance": 8.2e8},
    },
    "NODE_xyz": {
        "age":    {"mean": 44.8, "count": 98, "variance": 110.3},
        "income": {"mean": 52000, "count": 98, "variance": 7.9e8},
    },
}
```

!!! note "FAResult leaf structure"
    Each leaf is a dict of raw stat primitives. Aggregating the mean, for example, requires both `mean` and `count` (to compute a weighted average), which is why each leaf stores multiple keys even when you only asked for one statistic.

### Computable vs available stats

- `available_stats()` — the raw stat keys present at the leaves (e.g. `["count", "mean", "variance"]`). These are what was actually computed on the nodes.
- `computable_stats()` — the higher-level statistics that can be derived from those raw keys (e.g. `["count", "mean", "std", "sum", "variance"]`). `std` is computable even if it was never sent by the nodes, because it can be derived from `mean`, `variance`, and `count`.

```python
print(result.available_stats())   # ['count', 'mean', 'variance']
print(result.computable_stats())  # ['count', 'mean', 'std', 'sum', 'variance']
```

### Accessing results

```python
# Globally aggregated value for one statistic — same structure as per-node output
result.global_stat("mean")
# → {'age': 45.8, 'income': 53200, ...}

# All computable stats merged into {column: {stat_name: value}}
result.global_stats()
# → {'age': {'mean': 45.8, 'std': 10.9, 'variance': 118.7, ...}, ...}

# Raw output from one node — useful for site-level analysis or debugging
result.node_stats("NODE_abc")

# Outputs from all nodes
result.all_node_stats()

# Inspect the output tree structure without raw values
result.schema
# → {'age': {}, 'income': {}, ...}

# Which nodes replied?
result.node_ids
```

### Methods summary

| Method / property | Description |
|---|---|
| `result.global_stat(stat_name)` | The globally aggregated value for one statistic (e.g. `"mean"`) |
| `result.global_stats()` | All computable stats aggregated and merged into `{column: {stat_name: value}}` |
| `result.node_stats(node_id)` | The raw statistics returned by a single node, before aggregation |
| `result.all_node_stats()` | Dict of all per-node raw outputs keyed by node ID |
| `result.computable_stats()` | List of statistic names that can be derived from the stored data |
| `result.available_stats()` | List of raw stat keys present in the stored output |
| `result.node_ids` | List of node IDs whose results are stored |
| `result.schema` | The output tree structure with leaf values replaced by `{}` — useful for programmatic inspection |

---

## Common Errors & Troubleshooting

| Error message | Cause | Fix |
|---|---|---|
| `At least one of 'stats' or 'stats_args' must be provided` | `fetch_stats()` was called with no arguments | Pass at least `stats="mean"` (or another statistic name) or a `stats_args` dict |
| `Federated Analytics are not allowed on this node` | A node has `allow_federated_analytics = False` in its config | Ask the node operator to enable the flag — see [Federated Analytics — Nodes](../nodes/federated-analytics.md) |
