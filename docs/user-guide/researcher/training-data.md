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

The method `training_data` is the method that should be defined while creating the training plan class. This method can 
take one input argument which is the `batch_size`. This argument represents the batch size that is going to be used for the 
data loader. Although this argument is not mandatory for `training_data`, if it exists,  each node will pass the value 
`batch_size` defined in the `training_args` while calling the `training_data` method. 

```python
from fedbiomed.common.training_plans import TorchTrainingPlan

class MyTrainingPlan(TorchTrainingPlan):
    def __init__(self):
        pass
        # ....
    def training_data(self, batch_size):
        pass
```

You can also remove the `batch_size` argument and define it inside the method. In this case, the `batch_size`defined in 
the training arguments will no longer have an impact. 

```python
def training_data(self):
    batch_size = 48
    pass
```

!!! note ""
    `training_data` can have a maximum of one argument which is `batch_size`. Since `training_data` is executed by the node and the nodes are aware of only one argument, 
     adding extra argument to this method will raise errors on the node side.


### Reading Datafiles 

As mentioned before, a node can store `IMAGE` and `CSV` dataset types. In this case, the way of loading these datasets 
might be different. The only information necessary for loading will be the file path where data files are stored. 
This path is accessible through `self.dataset_path`. If the dataset is a tabular `CSV` dataset  `self.dataset_path` 
will address to a file. Otherwise, if it is an `IMAGE` dataset, `self.dataset_path` will address to a directory where 
all the images are stored. Therefore, before loading the dataset it is important to know what type of dataset is going 
to be loaded. It is possible to send [list request to the nodes to get meta-data of the dataset there are 
deployed](../../user-guide/researcher/listing-datasets-and-selecting-nodes.md).

The following snippet shows an example of loading operation for a dataset of `CSV` type. 

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
    
    def training_data(self, batch_size):
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=',')
        X = dataset.iloc[:,0:15].values
        y = dataset.iloc[:,15]
        return DataManager(dataset=X, target=y.values, batch_size=batch_size)
```

It is also possible to use some model arguments in the training data method. For example, if 
the following model argument is passed to the model by the experiment. 

```python
model_args = {
    'feature_cols' : 15
}
```
`training_data` can be configured as follows:

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
        
    def training_data(self, batch_size):
        feature_cols = self.model_args()["feature_cols"]
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=',')
        X = dataset.iloc[:,0:feature_cols].values
        y = dataset.iloc[:,feature_cols]
        return DataManager(dataset=X, target=y.values, batch_size=batch_size)
```

### What `training_data` Should Return?

The method `training_data` should always return `DataManager` of Fed-BioMed defined in the module 
`fedbiomed.common.data.DataManager`. `DataManager` has been designed for managing different types of data objects for 
different types of training plans. It is also responsible for splitting a given dataset into training and validation if 
model validation is activated in the experiment. 

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

    def training_data(self, batch_size):
        feature_cols = self.model_args()["feature_cols"]
        dataset = pd.read_csv(self.dataset_path, header=None, delimiter=',')
        X = dataset.iloc[:,0:feature_cols].values
        y = dataset.iloc[:,feature_cols]
        return DataManager(dataset=X, target=y.values, batch_size=batch_size)
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

    def training_data(self, batch_size): 
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

    def training_data(self, batch_size):
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
def training_data(self, batch_size = 48):
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

Any keyword argument, except `dataset` and `target`, that was provided to the `DataManager` constructor in the 
`training_data` function will be passed as a keyword argument to the `DataLoader`. 
For PyTorch and scikit-learn experiments, the `DataLoaders` have been heavily inspired by the
[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 
class, so please refer to that documentation for the meaning of the supported keyword arguments. For example:

```python
class MyTrainingPlan(SKLearnTrainingPlan):
    def training_data(self, batch_size):
        return DataManager(dataset=X, target=Y, 
                           # The following arguments will be passed to the DataLoader:
                           batch_size=batch_size, shuffle=True, drop_last=False)
```

#### The special case of batch size

As you may have noticed below, `batch_size` represents a special case since it can be provided as argument to the 
`training_data` function. 

!!! Note "Recommended approach"
    For `batch_size`, we recommend that you always include it is argument to the `training_data` function, and forward
    that value as keyworkd argument to the `DataManager`, as such:
    ```python
    class MyTrainingPlan(SKLearnTrainingPlan):
        def training_data(self, batch_size):
            return DataManager(dataset=X, target=Y, batch_size=batch_size)
    ```

It is not useful to set a default value for the `batch_size` argument, as it will be ignored in favour of Fed-BioMed's 
internal default value set in the `TrainingArgs` class. 

## Conclusion

`training_data` should be provided in each training plan. The way it is defined is almost the same for each framework's 
training plan as long as the structure of the datasets is simple. Since the method is defined by users, it provides 
flexibility to load and pre-process complex datasets distributed on the nodes. However, this method will be executed 
on the node side. Therefore, typos and lack of arguments may cause errors in the nodes even if it does not create any 
errors on the researcher side.

