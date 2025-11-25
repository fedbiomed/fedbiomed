---
title: The Method `training_data` in Training Plan
description: The method `training_data` of the training plan is needed to load and process the dataset deployed on the nodes.
             It is a method that has to be defined in every training plan class.
keywords: training data,training plan,fedbiomed, loading dataset in FL
---

# Loading Dataset for Training

Datasets in the nodes are saved on the disk. Therefore, before the training, each node should load these datasets from
the file system. Since the type of datasets (image, tabular, etc.) and the way of loading might vary from one to
another, the user (researcher) should define a method called `training_data`. The method `training_data` is mandatory
for each training plan (`TorchTrainingPlan` and `SKLearnTrainingPlan`). If it is not defined nodes will return an error at
the very beginning of the first round.

## Defining The Method Training Data

The method `training_data` defines the logic related to loading data on the node. In particular,
it defines:

- the type of data and the `Dataset` class
- any preprocessing (either data transforms, imputation, or augmentation)
- also implicitly defines the `DataLoader` for iterating over the data

This method takes no inputs and returns a [`DataManager`](./training-data.md#the-datamanager-return-type),
therefore its signature is:

```python
def training_data(self) -> fedbiomed.common.datamanager.DataManager
```

The `training_data` method is always part of the training plan, as follows:

```python
from fedbiomed.common.training_plans import TorchTrainingPlan

class MyTrainingPlan(TorchTrainingPlan):
    def __init__(self):
        pass
        # ....
    def training_data(self):
        pass
```

For details on how arguments are passed to the data loader, please refer to the section below
[Passing arguments to data loaders](./training-data.md#passing-arguments-to-data-loaders).

### The `DataManager` return type

The method `training_data` should always return `DataManager` of Fed-BioMed defined in the module
`fedbiomed.common.datamanager.DataManager`. `DataManager` has been designed for managing different types of data objects for
different types of training plans. It is also responsible for splitting a given dataset into training and validation subsets if
model validation is activated in the experiment.

!!! info "What is a `DataManager`?"
    A `DataManager` is a Fed-BioMed concept that makes the link between a `Dataset` and the corresponding `DataLoader`.
    It has a generic interface that is framework-agnostic (Pytorch, sklearn, etc...)

`DataManager` takes two main input arguments as `dataset` and `target`.
In most cases, `dataset` should be an instance of one of PyTorch `Dataset`, and `target` should be `None`.

For handling backwards compatibility and some simple cases, a user does not have to instantiate a `Dataset` object.
User can pass the argument `dataset` as
Numpy `ndarray`, `pd.DataFrame` or `pd.Series`. The argument `target` should then be an instance of one of Numpy `ndarray`,
`pd.DataFrame` or `pd.Series`. By default, the argument `target` is `None`. If `target` is `None` the data manager
considers that the `dataset` is an object that includes both input and target variables. This is the case where
the dataset is an instance of the PyTorch dataset. **If `dataset` is an instance of Numpy `Array` or Pandas `DataFrame`,
it is mandatory to provide the `target` variable.**

 For handling any arbitrary type of data, a
 user is also allowed to define a [`CustomDataset`](../datasets/custom-dataset.md), where the user directly writes how to `read` and `get_item` from the dataset. In this case, it is upto the user to whether to pass the targets through the `dataset` object, `target` variable or to not give targets (unsupervised) at all.

As it is mentioned, `DataManager` is capable of managing/configuring datasets/data-loaders based on the training plans
that are going to be used for training. This configuration is necessary since each training plan requires different
types of data loader/batch iterator, but it is handled by the framework and requires no user action.

## Defining Training Data in Different Training Plans

### Defining Training Data for PyTorch Based Training Plans

In the following code snippet using the classical syntax, a PyTorch-based training plan
returns a `DataManager` object instantiated with a `Dataset`, and `target` is unused (`None`)

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MnistDataset

class MyTrainingPlan(TorchTrainingPlan):
    def init_model(self):
        # ....
    def init_dependencies(self):
        # ....
    def init_optimizer(self):
        # ....

    def training_data(self):
        dataset = MnistDataset()
        loader_arguments = {'shuffle': True}
        return DataManager(dataset, **loader_arguments)

```

In the following code snippet using backwards compatible syntax,
`training_data` of PyTorch-based training plan returns a `DataManager` object instantiated
with `dataset` and `target` as `pd.Series`. Since PyTorch-based training requires a PyTorch `DataLoader`, `DataManager`
converts `pd.Series` to a proper `torch.utils.data.Dataset` object and create a PyTorch `DataLoader` to pass it to the
training loop on the node side.

```python
import pandas as pd
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager

class MyTrainingPlan(TorchTrainingPlan):
    def init_model(self):
        # ....
    def init_dependencies(self):
        # ....
    def init_optimizer(self):
        # ....

    def training_data(self):
        feature_cols = self.model_args()["feature_cols"]
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=',')
        X = dataset.iloc[:,0:feature_cols].values
        y = dataset.iloc[:,feature_cols]
        return DataManager(dataset=X, target=y.values, )
```

It is also possible to define a custom PyTorch `Dataset` and use it in the `DataManager` without declaring the argument
`target`.

```python
import pandas as pd
from torch.utils.data import Dataset
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import CustomDataset

class MyTrainingPlan(TorchTrainingPlan):

    class CelebaDataset(CustomDataset):
        """Any custom dataset should inherit from the CustomDataset class"""

        # we dont load the full data of the images, we retrieve the image with the get item.
        # in our case, each image is 218*178 * 3colors. there is 67533 images. this take at least 7G of ram
        # loading images when needed takes more time during training but it won't impact the ram usage as much as loading everything

        def read(self):
            self.input_file = pd.read_csv(dataset_path,sep=',',index_col=False)
            x_train = self.input_file.iloc[:,0:features].values
            y_train = self.input_file.iloc[:,features].values
            self.X_train = torch.from_numpy(x_train).float()
            self.Y_train = torch.from_numpy(y_train).float()

        def get_item(self, index):
            return self.X_train[idx], self.Y_train[idx]

        def __len__(self):
            return len(self.Y_train)

    def init_dependencies(self):
        """Custom dataset and other dependencies"""
        deps = [
            "from fedbiomed.common.dataset import CustomDataset"
        ]
        return deps

    def training_data(self):
        feature_cols = self.model_args()["feature_cols"]
        dataset = self.CSVDataset(self.dataset_path, feature_cols)
        loader_kwargs = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset=dataset, **loader_kwargs)

```
In the code snippet above, `loader_kwargs` contains the arguments that are going to be used while creating a
PyTorch `DataLoader`.

### Defining Training Data for SkLearn Based Training Plans

The operations in the `training_data` for SkLearn based training plans are not much different than`TorchTrainingPlan`.

In the following code snippet using the classical syntax, a SkLearn-based training plan
returns a `DataManager` object instantiated with a `Dataset`, and `target` is unused (`None`)

```python
from fedbiomed.common.training_plans import FedPerceptron
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MnistDataset

class SkLearnClassifierTrainingPlan(FedPerceptron):
    def init_dependencies(self):
        # ....

    def training_data(self):
        ...

        dataset = MnistDataset(transform=...)

        return DataManager(dataset=dataset, shuffle=False)
```

In the following code snippet using backwards compatible syntax,
`training_data` of SkLearn training plan returns a `DataManager` object instantiated
with `dataset` and `target` as `pd.Series`.

```python
import pandas as pd
from fedbiomed.common.training_plans import FedPerceptron
from fedbiomed.common.datamanager import DataManager

class SGDRegressorTrainingPlan(FedPerceptron):
    def init_dependencies(self):
        # ....

    def training_data(self):
        num_cols = self.model_args()["number_cols"]
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=',')
        X = dataset.iloc[:,0:num_cols].values
        y = dataset.iloc[:,num_cols]
        return DataManager(dataset=X, target=y.values, batch_size)

```

## Preprocessing for Data

Since the method `training_data` is defined by the user, it is possible to do preprocessing before creating the
`DataManager` object. In the code snippet below, a preprocess for normalization is shown for the dataset MNIST.

```python
from fedbiomed.common.dataset import MnistDataset

def training_data(self):
    # Custom torch Dataloader for MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_mnist = MnistDataset(transform=transform, target_transform=transform)
    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    return DataManager(dataset=dataset_mnist, **train_kwargs)
```

!!! info ""
    Training and validation partitions are created on the node side using returned `DataManager` object. Therefore,
       preprocessing in `training_data` will be applied for both validation and training data.

## Data Loaders

A `DataLoader` is an iterable class that takes care of handling the logic of iterating over a certain dataset.
Thus, while a `Dataset` is concerned with loading and preprocessing samples one-by-one, the `DataLoader` is responsible
for:

- calling the dataset's `__getitem__` method when needed
- collating samples in a batch
- shuffling the data at every epoch
- in general, managing the actions related to iterating over a certain dataset

`DataLoader` is handled internally, so user does not instantiate it.

### Passing arguments to Data Loaders

!!! important ""
    All of the key-value pairs contained in the `loader_args` sub-dictionary of
    [`training_args`](./experiment.md#batch-size-and-other-loader-arguments)
    are passed as keyword arguments to the data loader.
    Additionally, any keyword arguments passed to the `DataManager` class inside the
    `training_data` function are also passed as keyword arguments to the data loader.

For example, the following setup:

```python
class MyTrainingPlan(TorchTrainingPlan):
    # ....
    def training_data(self):
        dataset = MyDataset()
        return DataManager(dataset, shuffle=True)

training_args = {
    'loader_args': {
        'batch_size': 5,
        'drop_last': True
    }
}
```

Leads to the following data loader definition:

```python
loader = DataLoader(dataset, shuffle=True, batch_size=5, drop_last=True)
```

!!! warning "Double-specified loader arguments"
    Keyword arguments passed to the `DataManager` class take precedence over arguments with the same name provided
    in the `loader_args` dictionary.

For PyTorch and scikit-learn experiments, the `DataLoaders` have been heavily inspired by the
[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
class, so please refer to that documentation for the meaning of the supported keyword arguments.

## Data Format in Training Plan

Training and testing data returned by the `DataLoader` is handled to the training plan in a framework-specific way:

* for PyTorch training plan, data is received via the `data` and `target` arguments of the `training_step` / `testing_step` methods
  of the training plan
* for SkLearn training plan, data is directly consumed to train and test the ML model (no user method)

```python
class MyTrainingPlan(TorchTrainingPlan):
    # ...
    def training_step(self, data, target):
        # ....
    def testing_step(self, data, target):
        # ....
```

This paragraph describes the format of the training plan data batches.
For details about the data format preprocessing refer to the [Applying Transformations](../datasets/applying-transformations.md) documentation.


Training plan data format and preprocessing concerns the user for 2 main reasons:

* know which format are `data` and `target` in `training_step` and `testing_step`, in the case of PyTorch training plan
* know which format needs to be delivered to the `DataManager` in the `training_data` function, for each training plan framework

### Batches of data sample

`data` and `target` are delivered as **batches** to the training plan. Each batch is a list of (at most) `batch_size` data samples.

Each data sample is composed of a `data` and a `target` **entry**.
If `target` is `None`, then it means there is no target in this dataset's data samples, for example for unsupervised learning.

!!! warning "Target is needed for PyTorch"
    In current Fed-BioMed version, `target` is needed for PyTorch training plan (it cannot be `None`)
    Workaround for dataset without target data is to add dummy target values in the dataset.

| Framework | data != None | target != None | target == None |
| --------- | ------------ | -------------- | -------------- |
| PyTorch   | ✅ | ✅ | :x: |
| SkLearn   | ✅ | ✅ | ✅ |

### Entries structure and modalities in a data sample

Then each `data` or `target` entry needs to be:

* either monomodal: a data **structure** as expected by the framework
* or multimodal: a **dict** indexed by `str` (name of the modality) where each value is a data structure as expected by the framework

| Framework | data structure | dict of data structure |
| --------- | ------------ | -------------- |
| PyTorch   | `torch.Tensor` | `Dict[str, torch.Tensor]` |
| SkLearn   | `np.ndarray` | `Dict[str, np.ndarray]` with 1 modality |

!!! info "SkLearn is monomodal in Fed-BioMed"
    SkLearn support in Fed-BioMed is monomodal, to match the behaviour of most SkLearn functions.
    The `Dict` syntax is supported for convenience, but
    only **one** modality can exist in the `Dict`. To handle multimodal data in SkLearn,
    use a transformation to merge inputs from multiple modalities in a single entry.

In SkLearn, most functions require vectors (1-dimensional inputs) or values (0-dimensional inputs).
To match this behaviour, `DataLoader` checks that the input `np.ndarray` passed to the SkLearn training plan
has `ndim =< 1`.

!!! info "SkLearn training plan needs `ndim =< 1`
    If the `Dataset` returns `np.ndarray` with `ndim >= 2`, use a dataset transformation to flatten
    the data structure, as explained in
    [Applying Transformations](../datasets/applying-transformations.md) documentation.

### Format of structure values

Finally, each structure (`torch.Tensor`, `np.ndarray`) has values of one **type** (`dtype`): a `float`, an `int`.
For each framework, each type has a preferred **format** (eg `float32`, `float64`).

Fed-BioMed automatically assigns the preferred format for the framework before delivering the values to the training plan.

| Framework | value type | `dtype` format |
| --------- | ---------- | -------------- |
| PyTorch   | any floating point | `torch.get_default_dtype()` usually `torch.float32` |
| PyTorch   | any integer        | `torch.long` |
| scikit-learn | any `np.floating`  | `np.float64` |
| scikit-learn | any `np.integer`   | `np.int64`   |

For example, for `PyTorch`, floating point values are delivered as `torch.get_default_dtype()` which is usually (and by default) `torch.float32`


## Conclusion

`training_data` should be provided in each training plan. The way it is defined is almost the same for each framework's
training plan as long as the structure of the datasets is simple. Since the method is defined by users, it provides
flexibility to load and pre-process complex datasets distributed on the nodes. However, this method will be executed
on the node side. Therefore, typos and lack of arguments may cause errors in the nodes even if it does not create any
errors on the researcher side.

