# Using `CustomDataset` in Federated Workflows

## Introduction

`CustomDataset` is a flexible base class for creating custom datasets in Fed-BioMed. It enforces a clear structure for user-defined datasets, ensuring compatibility with federated learning workflows defined on the node side. Users should implement their own data loading and sample retrieval logic by inheriting from `CustomDataset`.

## Principles 

- `CustomDataset` enforces the implementation of `read`, `get_item`, and `__len__` methods in subclasses.
- You should avoid overriding `__getitem__` and `__init__`; `CustomDataset` will raise an error if these methods are overwritten, as they are reserved for internal use.
- Ensure that `get_item` returns a tuple (e.g., `(data, target)`); `CustomDataset` automatically checks that this method returns a tuple.
- `CustomDataset` must respect the data format required by the training plan. `get_item` should return `data` and `target` in the format expected by the framework used for training (e.g., `np.ndarray` for scikit-learn, `torch.Tensor` for PyTorch).

## How to Use

### Subclassing

To create your own dataset class, inherit from `CustomDataset`. The `CustomDataset` provides a `path` attribute, which points to either a folder or a specific file. This makes it easy to use the path as the root location for datasets that include multiple types of data (multi-modality), such as images and text.

!!! note "Real-World Deployment"
    In real-world use with Fed-BioMed, if you want to use a dataset type that isn’t already supported, you’ll need to know whether the `path` refers to a file or a folder. This information is included in the dataset description, which you can retrieve using the `list` or `search` methods of the `Experiment` class.

    **Key points:**
        - Inherit from `CustomDataset` to make your own dataset class.
        - The `path` attribute tells you where your data is stored (file or folder).
        - For new dataset types, check the dataset description for details about the path.
        - Use  `list` and `search` queries to get dataset descriptions. Please see the documentation for [listing datasets](../researcher/listing-datasets-and-selecting-nodes.md)

`read(self)` has to be written in the subclass. This method is called once. It should read the data from the file system and loaded in the correct format.

`get_item(self, idx)` is responsible for getting a single sample from the dataset. It should return a tuple `(data, target)` for the given index. For analytics or non-target studies it can return `data, None`.  The return format should follow the framework-specific convention (e.g., `torch.Tensor` for PyTorch and `numpy.ndarray` for scikit-learn). For a complete list of supported formats, see `fedbiomed.common.data_type.DataReturnFormat`."

- `__len__(self)` should returns the number of samples in the dataset.

!!! important "Special methods"
    In order to avoid unexpected errors, do not override `__getitem__` or `__init__` directly

### Example: CSV Dataset

```python
import pandas as pd
import torch
from fedbiomed.common.dataset._custom_dataset import CustomDataset

class CsvDataset(CustomDataset):
    def read(self):
        """Reads the data"""
        self.input_file = pd.read_csv(self.path, sep=',', index_col=False)
        x_train = self.input_file.loc[:, ['col-1', 'col-2', 'col-4', 'target-col']].values
        y_train = self.input_file.loc[:, 'target-col'].values
        self.X_train = torch.from_numpy(x_train)
        self.Y_train = torch.from_numpy(y_train)

    def __len__(self):
        """Returns the sample size"""
        return len(self.Y_train)

    def get_item(self, idx):
        """Gets single sample from the dataset"""
        return self.X_train[idx], self.Y_train[idx]
```
## Error Handling

If you do not implement the required methods (`read`, `get_item`, `__len__`), or if `get_item` does not return a tuple, a `FedbiomedError` will be raised on the node side before training begins. You will need to update your dataset class to fix these issues.Therefore, it is recommended to develop and test your custom dataset class locally before deploying it for federated training.

## Best Practices

- Use the `path` attribute to specify the dataset location.
- Validate your data types to match the expected format.

## Transformation
Transformations can be implemented in either the `read` or `get_item` methods. Unlike Fed-BioMed's provided dataset classes, custom datasets **do not accept** `transform` or `target_transform` arguments.

## Summary

`CustomDataset` provides a robust and structured way to integrate custom data sources into Fed-BioMed. By following the required subclassing pattern, you ensure your dataset is compatible and ready for federated learning tasks.
