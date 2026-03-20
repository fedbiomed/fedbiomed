# Federated Analytics — Datasets

## Overview

**Federated Analytics (FA)** lets researchers compute statistics — such as means, variances, or histograms — across datasets that live on multiple remote **nodes**, all **without the raw data ever leaving those nodes**.

In FedBioMed, a *node* is a machine controlled by a data owner (e.g. a hospital) that holds a local dataset. Instead of centralising data, each node computes statistics locally and sends only the aggregated summaries back to the researcher.

This page covers the **dataset side** of Federated Analytics: which datasets support it and how to make a custom dataset FA-compatible.

> - For how to **run** analytics as a researcher, see [Federated Analytics — Researcher](../researcher/federated-analytics.md).
> - For how to **enable** FA on a node, see [Federated Analytics — Nodes](../nodes/federated-analytics.md).

---

## What FA can compute — and on which datasets

### Dataset element types

FA treats every dataset as a collection of samples, and each sample as one or more **elements**. Each element has a type that determines which statistics can be computed on it:

| Element type | What it represents |
|---|---|
| **ROW** | A single row of named columns (tabular data) |
| **IMAGE** | An N-dimensional array without named columns |

---

!!! note "Dataset element types"
    - Multi-modal datasets can contain both types simultaneously — FA handles them independently.
    - Built-in [dataset classes](index.md) (`TabularDataset`, `MedicalFolderDataset`) declare their element type automatically and are FA-compatible out of the box. If you are writing a custom dataset you must declare it yourself — see below.

### Available statistics

The statistics available depend on the element type:

---

**ROW elements (tabular data)**

| Statistic | What it computes | Extra arguments required |
|---|---|---|
| `count` | Number of non-missing values per column | — |
| `mean` | Weighted mean per column across all nodes | — |
| `variance` | Population variance per column across all nodes | — |
| `histogram` | Per-column frequency counts in fixed bins | `bin_edges` per column (must be identical on all nodes) |

---

**IMAGE elements**

| Statistic | What it computes | Extra arguments required |
|---|---|---|
| `count` | Number of samples and shape information | — |
| `mean` | Element-wise mean across all nodes | — |
| `variance` | Element-wise population variance across all nodes | — |

---

## Making a Custom Dataset FA-Compatible

If your dataset inherits from `fedbiomed.common.dataset.Dataset`, one addition is needed.

!!! note "Data return format"
    The analytics engine requires your dataset to return data as NumPy arrays (`DataReturnFormat.SKLEARN`). Make sure this is set during your dataset's initialisation before using FA.

### Implement `analytics_schema()`

`analytics_schema()` returns a **description of what `__getitem__` produces**, so the analytics engine knows how to interpret each sample.

`__getitem__` returns a `(data, target)` tuple. `analytics_schema()` mirrors that structure: it returns a `(data_spec, target_spec)` tuple where each spec is either a `RowSpec`, an `ImageSpec`, or `None` (meaning "skip this part").

- Use `RowSpec(columns=[...])` when the element is a 2-D NumPy array whose columns have names (tabular data). The column list must match the column order that `__getitem__` actually returns in that position.
- Use `ImageSpec()` when the element is an N-D NumPy array.
- Use `None` for parts that analytics should ignore (typically the target).

```python
from fedbiomed.common.dataset_types import RowSpec, ImageSpec

class MyDataset(Dataset):
    def __getitem__(self, idx):
        # returns (array with columns ["age", "weight"], label)
        ...

    def analytics_schema(self):
        # Mirrors __getitem__: describe `data` with RowSpec, skip `target` with None
        return RowSpec(columns=["age", "weight"]), None
```

For a **multi-modal dataset** whose `__getitem__` returns a `dict` as its first element, the schema's first element must be a matching `dict` — same keys, each mapped to the appropriate spec:

```python
    def __getitem__(self, idx):
        # data is a dict; keys must match the schema below
        data = {
            "demographics": array_of_shape(n, 2),   # tabular
            "T1": array_of_shape(h, w, d),           # 3-D image
        }
        return data, None

    def analytics_schema(self):
        return {
            "demographics": RowSpec(columns=["age", "weight"]),
            "T1": ImageSpec(),
        }, None
```

---

## Common Errors & Troubleshooting

| Error message | Cause | Fix |
|---|---|---|
| `Dataset does not implement 'analytics_schema'` | Custom dataset is missing the `analytics_schema()` method | Add `analytics_schema()` (see [above](#2-implement-analytics_schema)) |
| `Dataset format … is not supported for analytics` | The dataset is not configured to return NumPy arrays | Ensure `self.to_format = DataReturnFormat.SKLEARN` is set during dataset initialisation |
| `Dataset does not support analytics method 'compute_stats'` | The dataset class does not inherit from `fedbiomed.common.dataset.Dataset` | Make sure your class inherits from `fedbiomed.common.dataset.Dataset` |
