---
title: The Training Plan of Fed-BioMed
description: A training plan is a class that defines the federated model training. It is responsible for providing base methods which allow every node to perform the training process.
keywords: training data,training plan,fedbiomed
---

# The Training Plan

A training plan is a class that defines the four main components of federated model training: the data, the model, the loss and the optimizer.
It is responsible for providing custom methods allowing every node to perform the training. 
In Fed-BioMed, you will be required to define a training plan class before submitting a federated training experiment. 
You will do so by sub-classing one of the base training plan classes provided by the library, 
and overriding certain methods to suit your needs as explained below.
The code of the whole training plan class is shipped to the nodes, meaning that
you may define custom classes and functions inside it, and re-use them within the training routine. 

!!! abstract "Training Plans"
    A Training Plan contains the recipe for executing the training loop on the nodes. It defines: the data, the model,
    the loss function, and the optimizer. The code in the training plan is shipped in its entirety to the nodes, where
    its different parts are executed at different times during the training loop.

## The `TrainingPlan` class

Fed-BioMed provides a base training plan class for two commonly-used ML frameworks: PyTorch (`fedbiomed.common.training_plans.TorchTrainingPlan`) 
and scikit-learn (`fedbiomed.common.training_plans.SKLearnTrainingPlan`). Therefore, the first step of the definition of your
federated training experiment will be to define a new training plan class that inherits from one of these. 

### Pytorch Training Plan
The interfaces for the two frameworks differ quite a bit, so let's start by taking the example of PyTorch:

```python
from fedbiomed.common.training_plans import TorchTrainingPlan


class MyTrainingPlan(TorchTrainingPlan):
    pass
```

The above example will not lead to a meaningful experiment, because we need to provide at least the following information
to complete our training plan:

- a model instance
- an optimizer instance
- a list of dependencies (i.e. modules to be imported before instantiating the model and optimizer)
- how to load the training data (and potential preprocessing)
- a loss function

Following the PyTorch example, here is what the prototype of your training plan would look like:
```python
from fedbiomed.common.training_plans import TorchTrainingPlan

class MyTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        # defines and returns a model
        pass

    def init_optimizer(self, optimizer_args):
        # defines and returns an optimizer
        pass

    def init_dependencies(self):
        # returns a list of dependencies
        pass

    def training_data(self):
        # returns a Fed-BioMed DataManager object
        pass
    
    def training_step(self, data, target):
        # returns the loss
        pass
``` 

### Scikit-learn Training Plan

In the case of scikit-learn, Fed-BioMed already does a lot of the heavy lifting for you by providing the
`FedPerceptron`, `FedSGDClassifier` and `FedSGDRegressor` classes as training plans. These classes already take care
of the model, optimizer, loss function and related dependencies for you, so you only need to define how the data will
be loaded. For example, in the case of `FedSGDClassifier`:

```python
from fedbiomed.common.training_plans import FedSGDClassifier

class MyTrainingPlan(FedSGDClassifier):
    def training_data(self):
        # returns a Fed-BioMed DataManager object
        pass
```

### 

!!! warning "Definition of `__init__` is discouraged for all training plans"
    As you may have noticed, none of the examples defined an `__init__` function for the training plan. This is on
    purpose! Overriding `__init__` is not required, and is actually discouraged, as it is reserved for
    the library's internal use.
    If you decide to override it, you do it at your own risk!


## Accessing the Training Plan attributes

Fed-BioMed provides the following getter functions to access Training Plan attributes:

| attribute           | function           | TorchTrainingPlan  | SKLearnTrainingPlan | notes | 
|---------------------|--------------------|--------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model               | `model()`          | :heavy_check_mark: | :heavy_check_mark:  | you may not dynamically reassign a model. The instance of the model is created at initialization by storing the output of the `init_model` function.              |
| optimizer           | `optimizer()`      | :heavy_check_mark: | :x:                 | you may not dynamically reassign an optimizer. The instance of the optimizer is created at initialization by storing the output of the `init_optimizer` function. |
| model arguments     | `model_args()`     | :heavy_check_mark: | :heavy_check_mark:  | |
| training arguments  | `training_args()`  | :heavy_check_mark: | :heavy_check_mark:  | |
| optimizer arguments | `optimizer_args()` | :heavy_check_mark: | :x:                 | |

####

!!! warning "Lifecycle of Training Plan Attributes"
    The attributes in the table above will not be available during the `init_model`, `init_optimizer` and 
    `init_dependencies` functions, as they are set just after initialization. You may however use them in the definition
    of `training_data`, `training_step` or `training_routine`.

## Defining the training data

The method `training_data` defines how datasets should be loaded in nodes to make them ready for training.
In both PyTorch and scikit-learn training plans, you are required to define a `training_data` method with the following
specs:

1. takes as input a `batch_size` parameter
2. returns a `fedbiomed.common.data.DataManager` object
3. inside the method, a dataset is instantiated according to the data type that you wish to use (one of `torch.Dataset`,
   `numpy.ndarray` or a `*Dataset` class from the `fedbiomed.common.data` module)
4. the dataset is used to initialize a `DataManager` class to be returned

The signature of the `training_data` function is then:
```python
def training_data(self) -> DataManager:
```

You can read the documentation for [training data](../../researcher/training-data) to
learn more about the `DataManager` class and various use cases.

## Initializing the model

In Pytorch training plans, you must also define a `init_model` function with the following signature:
```python
def init_model(self, model_args: Dict[str, Any]) -> torch.nn.Module:
```

The purpose of `init_model` is to return an instance of a trainable PyTorch model. Since the definition of such models 
can be quite large, a common pattern is to define the neural network class inside the training plan namespace, 
and simply instantiate it within `init_model`. This also allows to minimize the amount of adjustments needed to go
from local PyTorch code to its federated version. Remember that only the code defined inside the training plan
namespace will be shipped to the nodes for execution, so you may not use classes that are defined outside of it.

The Pytorch neural network class that you define must satisfy the following constraints:
1. it should inherit from `torch.nn.Module`
2. it should implement a `forward` method that takes a `torch.Tensor` as input and returns a `torch.Tensor` 
Note that inheriting from `torch.nn.Sequential` and using the default `forward` method would also respect the
conditions above. 

The `model_args` argument is a dictionary of model arguments that you may provide to the `Experiment` class and that
will be automatically passed to the `init_model` function internally. If you followed the suggested pattern of defining
the model class within the training plan namespace, you can easily adapt the model's constructor to make use of any
model arguments that you wish to define.

The example below, adapted from our getting started notebook, shows the suggested pattern, the use of `init_model`, and
the use of `model_args`.

```python
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager


# Here we define the model to be used. 
# You can use any class name (here 'Net')
class MyTrainingPlan(TorchTrainingPlan):
    
    # Defines and return model 
    def init_model(self, model_args):
        return self.Net(model_args = model_args)
    
    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            
            fc_hidden_layer_size = model_args.get('fc_hidden_size', 128)
            
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, fc_hidden_layer_size)
            self.fc2 = nn.Linear(fc_hidden_layer_size, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            output = F.log_softmax(x, dim=1)
            return output

    def training_data(self):
        pass    
    
    def training_step(self, data, target):
        pass
    
    def init_optimizer(self, optimizer_args):
        pass

    def init_dependencies(self):
        pass
```

## Defining the optimizer

In Pytorch training plans, you must also define a `init_optimizer` function with the following signature:
```python
def init_optimizer(self, optimizer_args: Dict[str, Any]) -> torch.optim.Optimizer:
```

The purpose of `init_optimizer` is to return an instance of a PyTorch optimizer. You may instantiate a "vanilla" 
optimizer directly from `torch.optim`, or follow a similar pattern to `init_model` by defining a custom optimizer class 
within the training plan namespace. 

####

!!! info "The output of `init_optimizer` must be a `torch.optim` type"
    The output of `init_optimizer` must be either a vanilla optimizer provided by the `torch.optim` module, or a class
    that inherits from `torch.optim.Optimizer`

Similarly, the `optimizer_args` follow the same pattern as `model_args` described above. 
Note that the learning rate will always be included in the optimizer arguments with the key `lr`.

A pretty straightforward example can be again found in the getting started notebook
```python
def init_optimizer(self, optimizer_args):
    return torch.optim.Adam(self.model().parameters(), lr = optimizer_args["lr"])
```

## Defining the loss function

The PyTorch training plan requires you to define the loss function via the `training_step` method, with the following
signature:
```python
def training_step(self, data, target) -> float:
```

The `training_step` method of the training class defines how the cost is computed by forwarding input values through the 
network and using the loss function. It should return the loss value. By default, it is not defined in the parent 
`TrainingPlan` class: it should be defined by the researcher in his/her model class, same as the `forward` method.  
An example of training step for PyTorch is shown below.

```python
    def training_step(self, data, target):
        output = self.forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss
```

### Type of `data` and `target`
The `training_step` function takes as input two arguments, `data` and `target`, which are obtained by cycling through
the dataset defined in the `training_data` function. There is some flexibility concerning what type of variables they
might be. 

In a Pytorch training plan, the following data types are supported: 

- a `torch.Tensor`
- a collection (a `dict`, `tuple` or `list`) of `torch.Tensor`
- a recursive collection of collections, arbitrarily nested, that ultimately contain `torch.Tensor` objects

!!! warning "Be aware of the data types in your dataset"
    It is ultimately your responsibility to write the code for `training_step` that correctly handles the data types
    returned by the `__getitem__` function of the dataset you are targeting. Be aware of the specifics of your dataset
    when writing this function.

## Adding Dependencies

By dependencies we mean here the python modules that are necessary to build all the various elements of your training
plan on the node side.  
The method `init_dependencies` allows you to indicate modules that are needed by your model class, with the following 
signature:
```python
def init_dependencies(self) -> List[str]:
```

Each dependency should be defined as valid import statement in a string, for example `from torch.optim import Adam` or 
`import torch`. You must specify dependencies for any python module that you wish to use, regardless of whether it 
is for the data, optimizer, model, etc...

## `training_routine` 

The training routine is the heart of the training plan. This method performs the model training loop, based on given 
[model and training](../../researcher/experiment) arguments. For example, if the
model is a neural network based on the PyTorch framework, the training routine is in charge of performing the training 
part over looping epochs and batches. If the model is a Scikit-Learn model, it fits the model by the given ML method 
and Scikit-Learn does the rest. The training routine is executed by the nodes after they have received a train request
from the researcher and downloaded the training plan file.

!!! warning "Overriding `training_routine` is discouraged"
    Both PyTorch and scikit-learn training plans already implement a `training_routine`, that internally uses the
    `training_step` provided by you to compute the loss function (only in the PyTorch case). Overriding this default
    routine is strongly discouraged, and you may do so only at your own risk.

As you can see from the following code snippet, the training routine requires some training arguments such 
as `epochs`, `lr`, `batch_size` etc. Since the `training_routine` is already defined by Fed-BioMed, you are only allowed 
to control the training process by changing these arguments. Modifying the training routine from the training plan class might raise unexpected errors. These training arguments are passed to the node by
the [experiment](../../researcher/experiment) through the network component.

```python
 def training_routine(self,
                         epochs: int = 2,
                         log_interval: int = 10,
                         lr: Union[int, float] = 1e-3,
                         batch_size: int = 48,
                         batch_maxnum: int = 0,
                         dry_run: bool = False,
                         ... ):

        # You can see details from `fedbiomed.common.torchnn`
        # .....

        for epoch in range(1, epochs + 1):
            training_data = self.training_data()
            for batch_idx, (data, target) in enumerate(training_data):
                self.train() 
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                res = self.training_step(data, target)
                res.backward()
                self.optimizer.step()

                #.....
```


## Saving and Loading Model

Each training plan provides save and load functionality. These are required for loading and saving model parameters 
into the file system after or before the training in the nodes and the researcher part. Consequently, 
[experiment](../../researcher/experiment) can upload and download the model parameters. Indeed, each framework
has its own way to load and save models.

You can access these classes from the `fedbiomed/common` directory to see them in more detail. 

!!! warning "Overriding `load` and `save` is discouraged"
    Both PyTorch and scikit-learn training plans already implement `load` and `save`. Overriding this default
    routines is strongly discouraged, and you may do so only at your own risk.





