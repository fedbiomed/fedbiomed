# Advanced Optimization in Fed-BioMed

Advanced Optimization can be done in `Fed-BioMed` through the use of `declearn`, a Python package that provides gradient-based `Optimizers`. `declearn` is cross-machine learning framework, meaning that it can be used with most machine learning frameworks.

The following chapter explains in depth how to use `declearn` optimization feature in `Fed-BioMed`. For an example, please refer to the [Advanced Optimizer tutorial]()

## `Declearn` based Optimizer: a cross framework `Optimizer`

### Introduction: what is `declearn` package?

[`declearn` package](https://gitlab.inria.fr/magnet/declearn/declearn2) is another Federated Learning framework modular and combinable, providing state-of-the-art gradient-based `Optimizer` algorithms. In `Fed-BioMed`, we are only using its `Optimization` facility, leaving aside all other components of `declearn` that we don't use in `Fed-BioMed`.


**References**: For further details about `declearn`, you may visit
- [`declearn` repository](https://gitlab.inria.fr/magnet/declearn/declearn2)
- [`declearn` documentation](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/)


#### `declearn` interface in `Fed-BioMed`: the `Optimizer` object

In `Fed-BioMed`, we provide a `Optimizer` object, that works as an interface with `declearn` in order to use `declearn`'s Optimizers (see below `declearn`'s `OptiModules` and `declearn`'s `Regularizers`).

```
from fedbiomed.common.optimizers.optimizer import Optimizer
Optimizer(lr=.1, decay=.0, modules=[], regualrizers=[])
```

#### `declearn`'s `OptiModules`

`declearn` `OptiModules` are modules that convey `Optimizers`, which purpose is to optimize a loss function (that can be written using a PyTorch loss function or a scikit learn model). 

**Usage**:

 - For basic SGD (Stochastic Gradient Descent), we don't need to specify a `OptiModule` and a `Regularizer`
    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer

    lr = .01
    Optimizer(lr=.1)
    ```

- For a specfic Optimizer like Adam, we need to import `AdamModule` from `declearn` package. Hence,  it yields:

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.modules import AdamModule

    lr = .01
    Optimizer(lr=.1, modules=[AdamModule()])

    ```

For further information on `declearn OptiModule`, please visit [`declearn OptiModule`](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/api-reference/optimizer/Optimizer/) 


**List of available Optimizers provided by `declearn`**

To get a list of all available Optimizers in declearn, please enter (after having loaded `Fed-BioMed` conda environment)

```python


```

#### `declearn`'s `Regularizers`

#### Chaining `Optimizers` and `Regularizers` with `declearn` modules


#### How to use well-known Federated-Learning algorithm with `declearn` in `Fed-BioMed`?

## `declearn` optimizer on Node side


## `declearn` optimizer on Researcher side

## `declearn` auxiliary variables based `Optimizers`


### What is an auxiliary variable?


### An example using `Optimizer` with auxiliary variables: `Scaffold` with `declearn`


## Table to use common Federated Learning algorithm with `declearn` in `Fed-BioMed`

## Common Pitfalls using `declearn` Optimizers in `Fed-BioMed`

Below, we are summerizing all common pitfalls that may occur when using `declearn` package in `Fed-BioMed`
- `Optimization` on `Researcher` side is only possible through `declearn` Optimizers (and not through native Optimizer such as PyTorch Optimizers);
- Some `Optimizers` may requiere some synchronization: it is the case of `ScaffoldClientModule` and `ScaffoldServerModule`;
- For the moment `declearn` Optimizers that use `auxiliary variables` (such as `Scaffold`) is not compatible with `SecAgg`.

