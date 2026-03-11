# Federated Analytics

## Introduction

Federated Analytics (FA) lets researchers compute statistics across distributed node datasets **without the raw data ever leaving the nodes**. Each node computes partial statistics locally; the researcher aggregates them globally.

The feature is built on top of the standard dataset infrastructure: built-in dataset classes support FA out of the box. When writing a custom dataset, a small set of additions makes it FA-compatible.

---

## Node Configuration

Federated Analytics is **enabled by default** on every node. The relevant flag lives in the `[security]` section of the node configuration file:

```ini
[security]
allow_federated_analytics = True
```

To disable FA for a node, set the value to `False`. No other node-side change is required; once a dataset is registered with appropriate tags it is automatically eligible for analytics requests.

---

## Supported Datasets and Statistics

FA operates on two element types, each with its own set of supported statistics:

| Element type | Supported statistics |
|---|---|
| **ROW** (tabular / demographic data) | `count`, `mean`, `variance`, `histogram`* |
| **IMAGE** (NIfTI, pixel arrays, …) | `count`, `mean`, `variance` |

\* `histogram` requires `bin_edges` — passed via `stats_args` (see below).

All built-in dataset classes (`TabularDataset`, `MedicalFolderDataset`, image datasets) expose the correct element type automatically. Custom datasets that inherit from the base `Dataset` class need two additions:

1. **Return format** — set `self.to_format = DataReturnFormat.SKLEARN` so samples are served as NumPy arrays.
2. **Schema** — implement `analytics_schema()` returning a `RowSpec`, `ImageSpec`, or a `dict` of those for multi-modal data.

```python
from fedbiomed.common.dataset_types import DataReturnFormat, RowSpec, ImageSpec

class MyDataset(Dataset):
    def complete_initialization(self):
        self.to_format = DataReturnFormat.SKLEARN          # (1)

    def analytics_schema(self):                            # (2)
        return RowSpec(columns=["age", "weight"]), None    # or ImageSpec(), or a dict
```

---

## Researcher-Side Usage

FA is accessed through the `analytics` property of an `Experiment`:

```python
from fedbiomed.researcher.experiment import Experiment

exp = Experiment(tags=["#my-dataset"])
```

### Convenience Methods

Five one-liner methods cover the most common statistics:

```python
mean     = exp.analytics.mean()
variance = exp.analytics.variance()
count    = exp.analytics.count()
```

Each call returns the **globally aggregated scalar or structure** directly.

### Filtering Columns with `dataset_schema`

Pass a schema to restrict the computation to a subset of the dataset:

```python
# Only compute mean over 'age' and 'bmi'
mean = exp.analytics.mean(dataset_schema=["age", "bmi"])
```

For nested schemas (multi-modal datasets), pass a dict:

```python
mean = exp.analytics.mean(dataset_schema={"clinical": ["age", "bmi"]})
```

### Advanced: `fetch_stats`

`fetch_stats` gives full control and returns a raw `FAResult` for further inspection:

```python
result = exp.analytics.fetch_stats(
    stats="variance",
    dataset_schema=["age", "bmi"],
)

# Globally aggregated value for a single statistic
age_mean = result.global_stat("mean")

# All computable stats merged into one tree: {col: {stat: val}}
all_stats = result.global_stats()

# Raw per-node output (not aggregated)
node_output = result.node_stats("node-id-1")
```

### Statistics Requiring Arguments

`histogram` require extra arguments supplied via `stats_args`:

```python
# Histogram with explicit bin edges per column
result = exp.analytics.fetch_stats(
    stats_args={
        "histogram": {
            "age":    {"bin_edges": [0, 20, 40, 60, 80, 100]},
            "income": {"bin_edges": [0, 25000, 50000, 75000, 100000]},
        }
    }
)
```

### Caching Behaviour

Results are cached per `(node_ids, dataset_schema, stats_args)` combination. Requesting a statistic that is already present in the cache does **not** trigger a new network round-trip. The cache holds at most 32 result sets (FIFO eviction).

---

## FAResult Reference

| Method / property | Description |
|---|---|
| `result.global_stat(stat_name)` | Globally aggregated value for one statistic |
| `result.global_stats()` | All computable stats merged: `{col: {stat: val}}` |
| `result.node_stats(node_id)` | Raw output for a specific node (not aggregated) |
| `result.all_node_stats()` | Dict of all per-node raw outputs |
| `result.computable_stats()` | List of stats that can be aggregated from the stored data |
| `result.available_stats()` | List of stat keys present in the first node's output |
| `result.node_ids` | List of node IDs in the result |
| `result.schema` | Schema mirror of the output tree (leaves shown as `{}`) |

---

## Common Errors

| Error message | Cause | Fix |
|---|---|---|
| `Dataset does not implement 'analytics_schema'` | Custom dataset is missing the method | Add `analytics_schema()` |
| `Dataset format … is not supported for analytics` | `to_format` is `TORCH` instead of `SKLEARN` | Set `self.to_format = DataReturnFormat.SKLEARN` |
| `Dataset does not support analytics method 'compute_stats'` | Class does not inherit from `Dataset` | Inherit from `fedbiomed.common.dataset.Dataset` |
| `Federated Analytics are not allowed on this node` | Node `allow_federated_analytics = False` | Enable the flag in the node config |
| `At least one of 'stats' or 'stats_args' must be provided` | `fetch_stats()` called with no arguments | Pass at least one stat name or `stats_args` |
