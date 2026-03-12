# Federated Analytics — Researcher

## Overview

**Federated Analytics (FA)** lets you compute statistics — such as means, variances, or histograms — across datasets held by multiple remote nodes, without the raw data ever leaving those nodes.

The flow is:

1. You send a statistics request to all participating nodes.
2. Each node computes the requested statistics **locally** on its own data.
3. Each node sends back only the aggregated summary (not the raw data).
4. FedBioMed combines the per-node summaries into a single **global result** on your side.

This page covers the **researcher side**: how to issue analytics requests and interpret the results.

> - For which datasets support FA and how to make a custom dataset FA-compatible, see [Federated Analytics — Datasets](../datasets/federated-analytics.md).
> - For enabling FA on a node, see [Federated Analytics — Nodes](../nodes/federated-analytics.md).

---

## Entry Point: `Experiment` and its `analytics` object

`Experiment` is the single entry point for both federated training and federated analytics (see [Experiment](experiment.md) for more). 

```python
from fedbiomed.researcher.experiment import Experiment

exp = Experiment(tags=["#my-dataset"])

# exp.analytics is a FederatedAnalytics object, ready to use
print(exp.analytics.get_node_ids())   # nodes that have a matching dataset
```

You never construct `FederatedAnalytics` directly; it is always accessed through the `analytics` attribute of an experiment. All network communication, result caching, and aggregation are handled internally — from your perspective, you call a method and get back a result.

---

## Convenience Methods

The three most common statistics each have a one-liner that sends the request, waits for all nodes to respond, aggregates the results, and returns the final value directly:

```python
count    = exp.analytics.count()     # total number of samples across all nodes
mean     = exp.analytics.mean()      # per-column (or per-pixel) mean
variance = exp.analytics.variance()  # per-column (or per-pixel) variance
```

Each call returns the **globally aggregated result** — a structure matching your dataset's schema.

---

## The Dataset Schema

Every FA-compatible dataset declares a **schema** — a description of what its samples look like. The schema is what the dataset's `analytics_schema()` method returns: column names for tabular data, or nested structures for multi-modal datasets.

When you call `mean()`, `variance()`, or `fetch_stats()`, the analytics engine reads this schema from each node's dataset to know how to interpret each sample.

> You don't normally need to read or manipulate the schema yourself. It is used automatically. `dataset_schema` (described below) lets you select a subset of it.

---

## Filtering with `dataset_schema`

By default, statistics are computed over everything the schema describes. Pass `dataset_schema` to restrict computation to a subset of columns or modalities.

**Tabular dataset — select columns by name:**

```python
# Only compute the mean over 'age' and 'bmi' columns
mean = exp.analytics.mean(dataset_schema=["age", "bmi"])
```

The list must contain column names that exist in the dataset's schema.

**Multi-modal dataset — select modalities, and optionally columns within them:**

For datasets that expose multiple modalities (e.g. images paired with a clinical table), `dataset_schema` is either a list of modality names to include in full, or a dict mapping each modality name to a column selector for that modality (`None` means include all columns for that modality):

```python
# Include the whole 'T1' image modality and only 'age'/'bmi' from 'demographics'
mean = exp.analytics.mean(
    dataset_schema={"T1": None, "demographics": ["age", "bmi"]}
)
```

Any modality not mentioned in `dataset_schema` is excluded from the computation.

---

## `fetch_stats`

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

## Statistics Requiring Arguments

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

`stats_args` and `stats` can be combined in a single call. At least one of the two must be provided.

---

## Caching

To avoid redundant network requests, results are cached per `(node_ids, dataset_schema, stats_args)` combination. When you repeat a call with the same parameters, the cached result is returned immediately — no messages are sent to the nodes.

The cache holds at most 32 distinct result sets; the oldest entry is evicted when the limit is reached.

---

## FAResult Reference

`fetch_stats` returns an `FAResult` object. Its main methods are:

| Method / property | Description |
|---|---|
| `result.global_stat(stat_name)` | The globally aggregated value for one statistic (e.g. `"mean"`) |
| `result.global_stats()` | All computable stats aggregated and merged into `{column: {stat_name: value}}` |
| `result.node_stats(node_id)` | The raw statistics returned by a single node, before aggregation |
| `result.all_node_stats()` | Dict of all per-node raw outputs keyed by node ID |
| `result.computable_stats()` | List of statistic names that can be aggregated from the stored data |
| `result.available_stats()` | List of statistic keys present in the stored output (before checking aggregability) |
| `result.node_ids` | List of node IDs whose results are stored |
| `result.schema` | The structure of the output tree with leaf values replaced by `{}` — useful for programmatic inspection |

---

## Common Errors

| Error message | Cause | Fix |
|---|---|---|
| `At least one of 'stats' or 'stats_args' must be provided` | `fetch_stats()` was called with no arguments | Pass at least `stats="mean"` (or another statistic name) or a `stats_args` dict |
| `Federated Analytics are not allowed on this node` | A node has `allow_federated_analytics = False` in its config | Ask the node operator to enable the flag — see [Federated Analytics — Nodes](../nodes/federated-analytics.md) |