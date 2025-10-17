# Using `NativeDataset` in Federated Workflows

## Introduction

`NativeDataset` is the simplest way to use data that you do not want to register with the Node, and instead directly use with Fed-BioMed.

It is designed for cases such as you have a pre-determined dataset, such as a Torch Dataset, a NumPy array or a Python list. The main difference from the `CustomDataset` class is that `NativeDataset` **is not a wrapper for your subclass**, as you don't need to write a subclass to use it. 

## Principles

- **Bring your own data object:** Use any object that behaves like a dataset (e.g., a Torch `Dataset`, a NumPy array, or a Python list).
- **Provided in the `TrainingPlan`:** You create/pass this object **inside** the `training_data` method via `DataManager`.
- **No filesystem contract:** No `path` is required; `NativeDataset` assumes your data is already available to the process.
- **Supervised vs. unsupervised:** 
  - If your data source returns `(data, target)` pairs (e.g., most Torch datasets), it's treated as **supervised**.
  - If your source returns only `data` **(unsupervised)**, you should also provide a separate `target/label` sequence if your model needs labels.

!!! warning "Unsupervised scenario"
    In the **unsupervised** scenario, each item still return a tuple of `(data, None)`. Make sure your Model and your `training_step` function handles accordingly/


## Where `NativeDataset` fits in your workflow

1. **Define your `TrainingPlan`.**
2. **Inside** its `training_data` method, construct your data object(s) (Torch dataset, NumPy arrays, or lists).
3. Pass them to the **`DataManager`** that your plan returns.
4. Fed-BioMed uses those objects for local training on the node.

This approach is ideal when your data is already in memory, preprocessed, or supplied by another pipeline and you do not want the burden of additionally adding the dataset to your node.

## Supported Inputs (Conceptually)

- **PyTorch-style datasets** that implement `__len__` and `__getitem__` and return `(data, target)` tuples.
- **NumPy arrays**: feature matrix `X` and, optionally, a label vector/array `y` of the same length.
- **Python lists**:
  - A list of `(data, target)` tuples (supervised).
  - A list of `data` items plus a separate label list/array (supervised).
  - A list of `data` items (unsupervised tasks).

## Example Patterns

Below are minimal, **pattern-oriented** examples showing how to pass native data inside `training_data`. These examples focus on structure and placement—adapt the variable names and preprocessing to your project. 

### A) Using a PyTorch `Dataset`

```python
# inside your TrainingPlan class
def training_data(self):
    from torchvision import datasets

    # build dataset
    ds = datasets.MNIST(tensor_x, tensor_y)

    # Provide directly to the DataManager as a "native" dataset
    dm = DataManager(dataset=ds)  # (API specifics documented elsewhere)
    return dm
```

### B) Using NumPy arrays

```python
# inside your TrainingPlan class
def training_data(self):
    # X: np.ndarray of features, y: np.ndarray of labels (same length)
    # Ensure your model/training code expects NumPy-based batches
    dm = DataManager(dataset=X, target=y)
    return dm
```

### C) Using Python lists of pairs

```python
# inside your TrainingPlan class
def training_data(self):
    # pairs: list of (data, target) tuples
    pairs = [([1, 2, 3], 0), ([0, 1, 0], 1)]
    dm = DataManager(dataset=pairs)
    return dm
```

### D) Unsupervised data with separate targets (when needed)

```python
# inside your TrainingPlan class
def training_data(self):
    # data: list or np.ndarray of inputs
    # y: labels provided separately; must match length of data
    dm = DataManager(dataset=data, target=y)
    return dm
```

## Error Handling & Validation Hints

- **Length mismatches:** If you provide a separate target sequence, its length must match the data length.
- **Type/shape mismatches:** Ensure the provided data matches what your model and training step expects. 
- **Unsupervised training:** Ensure that your model and training step expects a tuple of `(data, None)` as a sample. 
- **Item access failures:** If your Torch-style dataset raises an error on indexing, fix it locally before integrating.

!!! note "Develop locally, then federate"
    Validate your native data object (and a small training loop) locally to catch shape, dtype, and batching issues early.
    Once it runs end-to-end, integrate it into your `TrainingPlan` and node workflow.

## Best Practices

- **Preprocess early:** Apply tokenization, normalization, or feature engineering *before* passing the dataset to `DataManager`.
- **Keep items lightweight:** If items are large, ensure batching works efficiently in your training loop.
- **Be explicit about labels:** For supervised tasks, prefer sources that return `(data, target)` pairs to reduce errors.

## Summary

`NativeDataset` streamlines federated training when your data is already prepared.

Instead of implementing a new on-disk dataset class, simply provide your Torch dataset, NumPy arrays, or Python lists **inside your `TrainingPlan`** through the `DataManager`. This keeps your workflow concise and focused on modeling, not I/O plumbing.
