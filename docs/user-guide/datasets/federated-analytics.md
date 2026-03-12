# Federated Analytics — Datasets

## Overview

**Federated Analytics (FA)** lets researchers compute statistics — such as means, variances, or histograms — across datasets that live on multiple remote **nodes**, all **without the raw data ever leaving those nodes**.

In FedBioMed, a *node* is a machine controlled by a data owner (e.g. a hospital) that holds a local dataset. Instead of centralising data, each node computes statistics locally and sends only the aggregated summaries back to the researcher.

This page covers the **dataset side** of Federated Analytics: which datasets support it and how to make a custom dataset FA-compatible.

> - For how to **run** analytics as a researcher, see [Federated Analytics — Researcher](../researcher/federated-analytics.md).
> - For how to **enable** FA on a node, see [Federated Analytics — Nodes](../nodes/federated-analytics.md).

---

## Supported Datasets and Statistics

FA classifies each dataset by its **element type** — the kind of data each sample contains:

| Element type | Typical use case | Supported statistics |
|---|---|---|
| **ROW** | Tabular / demographic data (one row = one sample) | `count`, `mean`, `variance`, `histogram`* |
| **IMAGE** | Medical images (NIfTI, pixel arrays, …) | `count`, `mean`, `variance` |

\* `histogram` requires column-specific bin edges supplied by the researcher at request time.

Built-in [dataset classes](index.md) (`TabularDataset`, `MedicalFolderDataset`) declare their element type automatically and are FA-compatible out of the box. If you are writing a custom dataset you must declare it yourself — see below.

---

## Making a Custom Dataset FA-Compatible

If your dataset inherits from `fedbiomed.common.dataset.Dataset`, two additions are needed.

### 1. Set the return format

The analytics engine expects data as NumPy arrays. Tell FA to use the NumPy format by setting `self.to_format` inside `complete_initialization` — the hook that FedBioMed calls after the base class sets up its internals:

```python
from fedbiomed.common.dataset_types import DataReturnFormat

class MyDataset(Dataset):
    def complete_initialization(self):
        self.to_format = DataReturnFormat.SKLEARN  # serve samples as NumPy arrays
```

> **Why `SKLEARN`?** The name comes from scikit-learn's convention of returning plain NumPy arrays rather than PyTorch tensors. It has nothing to do with using scikit-learn models — it simply means "give me a NumPy array".

### 2. Implement `analytics_schema()`

`analytics_schema()` returns a **description of what `__getitem__` produces**, so the analytics engine knows how to interpret each sample.

`__getitem__` returns a `(data, target)` tuple. `analytics_schema()` mirrors that structure: it returns a `(data_spec, target_spec)` tuple where each spec is either a `RowSpec`, an `ImageSpec`, or `None` (meaning "skip this part").

- Use `RowSpec(columns=[...])` when the element is a 2-D NumPy array whose columns have names (tabular data). The column list must match the column order that `__getitem__` actually returns in that position.
- Use `ImageSpec()` when the element is an N-D NumPy array without named columns (images, voxel grids, …).
- Use `None` for parts that analytics should ignore (typically the target).

```python
from fedbiomed.common.dataset_types import RowSpec, ImageSpec

class MyDataset(Dataset):
    def complete_initialization(self):
        self.to_format = DataReturnFormat.SKLEARN

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

## Common Errors

| Error message | Cause | Fix |
|---|---|---|
| `Dataset does not implement 'analytics_schema'` | Custom dataset is missing the `analytics_schema()` method | Add `analytics_schema()` (see [above](#2-implement-analytics_schema)) |
| `Dataset format … is not supported for analytics` | `to_format` is set to `TORCH` (tensors) instead of `SKLEARN` (NumPy) | Set `self.to_format = DataReturnFormat.SKLEARN` in `complete_initialization` |
| `Dataset does not support analytics method 'compute_stats'` | The dataset class does not inherit from `fedbiomed.common.dataset.Dataset` | Make sure your class inherits from `fedbiomed.common.dataset.Dataset` |
