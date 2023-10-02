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
for each training plan (`TrochTrainingPlan` and `SkLearnSGDModel`). If it is not defined nodes will return an error at 
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
def training_data(self) -> fedbiomed.common.data.DataManager
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
`fedbiomed.common.data.DataManager`. `DataManager` has been designed for managing different types of data objects for 
different types of training plans. It is also responsible for splitting a given dataset into training and validation if 
model validation is activated in the experiment. 

!!! info "What is a `DataManager`?"
    A `DataManager` is a Fed-BioMed concept that makes the link between a `Dataset` and the corresponding `DataLoader`.
    It has a generic interface that is framework-agnostic (Pytorch, sklearn, etc...)

`DataManager` takes two main input arguments as `dataset` and `target`. `dataset` should be an instance of one of PyTorch `Dataset`,
Numpy `ndarray`, `pd.DataFrame` or `pd.Series`. The argument `target` should be an instance of one of Numpy `ndarray`, 
`pd.DataFrame` or `pd.Series`. By default, the argument `target` is `None`. If `target` is `None` the data manager 
considers that the `dataset` is an object that includes both input and target variables. This is the case where 
the dataset is an instance of the PyTorch dataset. **If `dataset` is an instance of Numpy `Array` or Pandas `DataFrame`, 
it is mandatory to provide the `target` variable.** 

As it is mentioned, `DataManager` is capable of managing/configuring datasets/data-loaders based on the training plans 
that are going to be used for training. This configuration is necessary since each training plan requires different 
types of data loader/batch iterator. 

## Defining Training Data in Different Training Plans

### Defining Training Data for PyTorch Based Training Plans

In the following code snippet, `training_data` of PyTorch-based training plan returns a `DataManager` object instantiated 
with `dataset` and `target` as `pd.Series`. Since PyTorch-based training requires a PyTorch `DataLoader`, `DataManager` 
converts `pd.Series` to a proper `torch.utils.data.Dataset` object and create a PyTorch `DataLoader` to pass it to the 
training loop on the node side. 

```python
import pandas as pd
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager

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
from fedbiomed.common.data import DataManager

class MyTrainingPlan(TorchTrainingPlan):

    class CSVDataset(Dataset):
        """ Cusotm PyTorch Dataset """
        def __init__(self, dataset_path, features):
            self.input_file = pd.read_csv(dataset_path,sep=',',index_col=False)
            x_train = self.input_file.iloc[:,0:features].values
            y_train = self.input_file.iloc[:,features].values
            self.X_train = torch.from_numpy(x_train).float()
            self.Y_train = torch.from_numpy(y_train).float()

        def __len__(self):            
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

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
Currently, SkLearn based training plans do not require a data loader for training. This means that all samples will be 
used for fitting the model. That's why passing `**loader_args` does not make sense for SkLearn based training plans.
These arguments will be ignored even if they are set. 

```python
import pandas as pd
from fedbiomed.common.training_plans import FedPerceptron
from fedbiomed.common.data import DataManager

class SGDRegressorTrainingPlan(FedPerceptron):

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
def training_data(self):
    # Custom torch Dataloader for MNIST data
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
    dataset_mnist = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    return DataManager(dataset=dataset_mnist, **train_kwargs)
```

!!! info ""
    Training and validation partitions are created on the node side using returned `DataManager` object. Therefore, 
       preprocessing in `training_data` will be applied for both validation and training data.

## Data Loaders

A `DataLoader` is a class that takes care of handling the logic of iterating over a certain dataset. 
Thus, while a `Dataset` is concerned with loading and preprocessing samples one-by-one, the `DataLoader` is responsible
for:

- calling the dataset's `__getitem__` method when needed
- collating samples in a batch
- shuffling the data at every epoch
- in general, managing the actions related to iterating over a certain dataset

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
class, so please refer to that documentation for the meaning of the supported keyword arguments. For example:

## Conclusion

`training_data` should be provided in each training plan. The way it is defined is almost the same for each framework's 
training plan as long as the structure of the datasets is simple. Since the method is defined by users, it provides 
flexibility to load and pre-process complex datasets distributed on the nodes. However, this method will be executed 
on the node side. Therefore, typos and lack of arguments may cause errors in the nodes even if it does not create any 
errors on the researcher side.

