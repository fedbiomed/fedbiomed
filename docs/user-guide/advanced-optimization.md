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

```python
from fedbiomed.common.optimizers import Optimizer

Optimizer(lr=.1, decay=.0, modules=[], regualrizers=[])
```

#### `declearn`'s `OptiModules`

`declearn` `OptiModules` are modules that convey `Optimizers`, which purpose is to optimize a loss function (that can be written using a PyTorch loss function or a scikit learn model) in order to optimize a model. They should be imported from `declearn`'s `declearn.optimizer.modules` 

**Usage**:

 - For basic SGD (Stochastic Gradient Descent), we don't need to specify a `declearn` `OptiModule` and/or  a `Regularizer`
    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer

    lr = .01
    Optimizer(lr=lr)
    ```

- For a specfic Optimizer like Adam, we need to import `AdamModule` from `declearn` package. Hence,  it yields:

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.modules import AdamModule

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule()])

    ```
- It is possible to chain `Optimizer` with several `OptiModule`s, meaning to use several `Optimizer`s. Some chains of `OptiModule` may be non-sensical, so use it at your own risk! Below we showcase the use of Adam with Momentum

    ```python

    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.modules import AdamModule, MomentumModule

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule(), MomentumModule()])

    ```

For further information on `declearn OptiModule`, please visit [`declearn OptiModule`](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/api-reference/optimizer/Optimizer/) 


**List of available Optimizers provided by `declearn`**

To get a list of all available Optimizers in declearn, please enter (after having loaded `Fed-BioMed` conda environment)

```python


```

#### `declearn`'s `Regularizers`

`declearn`'s `Regularizers` are objects that enable the use of `Regularizer`, which add a additional term to the loss function one wants to Optimize through the use of optimizer. It mainly helps to get a more generalizable model, and prevents overfitting.

The optimization equaion yields to:

$w_{t+1} := w_t - \eta \nabla_k L(w) + \alpha \norm {w}$
`Regularizers` should be used with an Optimizer. For instance, SGD with Ridge regression, or Adam with Lasso regression. [`FedProx`]() is also considered as a regularization.

**Usage**:

- for example, **SGD with Ridge regression** will be writen as:

    ```python

    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.regularizers import RidgeRegularizer

    lr = .01
    Optimizer(lr=lr, regularizers=[RidgeRegularizer()])
    ```

- **Adam with Lasso Regression**:

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.modules import AdamModule
    from declearn.optimizer.regularizers import LassoRegularizer

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule()], regularizers=[LassoRegularizer()])

    ```
- Chaining several Regularizers: an example with Ridge and Lasso regularizers

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.regularizers import LassoRegularizer, RidgeRegularizer

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule(), MomentumModule()], regularizers=[LassoRegularizer(), RidgeRegularizer()])
    ```

For further information on `declearn Regularizer`, please visit [`declearn Regularzers`](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/api-reference/optimizer/regularizers/Regularizer/) 


#### Chaining `Optimizers` and `Regularizers` with `declearn` modules

It is possible in `declearn` to chain several `OptiModule`s and `Regularizers` in an Optimizer. 
Generaly speaking, `Optimizer` in `declearn` can be writen as:

$\theta_{t+1} := \theta_t - \eta\  \underbrace{ Opt( \vec{\nabla} f_{x,y} - \overbrace{ Reg(\theta)}^\textrm{\color{Red}Regularizer})}_\textrm{{\color{Green} OptiModule}} $

with 
$Opt : \textrm{OptiModule}$
$Reg : \textrm{Regularizer}$

When using several `OptiModule`s and `Regularizers` in an Optimizer, optimization equation becomes:


**Example**: let's write an `Optimizer` using `RMSProp` and `Momentum` `OptiModules`, and both Lasso and Ridge Regularizers.


```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from declearn.optimizer.modules import RMSPropModule, MomentumModule
    from declearn.optimizer.regularizers import LassoRegularizer, RidgeRegularizer

    lr = .01
    Optimizer(lr=lr,
              modules=[RMSPropModule(), MomentumModule()],
              regularizers=[LassoRegularizer(), RidgeRegularizer()])
```

#### How to use well-known Federated-Learning algorithm with `declearn` in `Fed-BioMed`?

## `declearn` optimizer on Node side

In order to use `declearn` to optimize `Node`s local model, you will have to edit `init_otpimizer` method in the `TrainingPlan`. Below we showcase how to use it with PyTorch framework


```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.optimizers.optimizer import Optimizer
from declearn.optimizer.modules import AdamModule
from declearn.optimizer.regularizers import RidgeRegularizer
...

class MyTrainingPlan(TorchTrainingPlan):
    ...

    def init_dependencies(self):
        deps = [
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from declearn.optimizer.modules import AdamModule",
                "from declearn.optimizer.regularizers import RidgeRegularizer"
                ]

        return deps
    
    def init_optimizer(self):
        return Optimizer(lr=.01, modules=[AdamModule()], regularizers=[RidgeRegularizer()])


```

!!! note "Important": you should specify the OptiModules imported in both the imports at the begining of the `Training Plan` as well as in the dependences (in the `init_dependencies` method in the `Training Plan`). The same holds for `declearn`'s `Regularizers`.

The same will hold for scikit-learn as shown below, using the same `Optimizer`:

```python
from fedbiomed.common.training_plans import FedSGDClassifier
from fedbiomed.common.optimizers.optimizer import Optimizer
from declearn.optimizer.modules import AdamModule
from declearn.optimizer.regularizers import RidgeRegularizer
...

class MyTrainingPlan(FedSGDClassifier):
    ...

    def init_dependencies(self):
        deps = [
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from declearn.optimizer.modules import AdamModule",
                "from declearn.optimizer.regularizers import RidgeRegularizer"
                ]

        return deps
    
    def init_optimizer(self):
        return Optimizer(lr=.01, modules=[AdamModule()], regularizers=[RidgeRegularizer()])

```
## `declearn` optimizer on Researcher side (`FedOpt`)

`Fed-BioMed` provides a way to use  **Adaptive Federated Optimization**, introduced as [`FedOpt` in this paper](https://arxiv.org/pdf/2003.00295.pdf). In the paper, authors considered the difference of the global model weights between 2 successive `Rounds` as a *pseudo gradient*, paving the way to `Optimizer`s on `Researcher` side.
To do so, `fedbiomed.researcher.experiment.Experiment` has a method to set the `Researcher Optimizer`: `Experiment.set_agg_optimizer` 

Below an example using the `set_agg_optimizer` with `FedYogi`:

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators import FedAverage
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from declearn.optimizer.modules import YogiModule as FedYogi

tags = ['#my-data']

exp = Experiment()
exp.set_training_plan_class(training_plan_class=MyTrainingPlan)
exp.set_tags(tags = tags)
exp.set_aggregator(aggregator=FedAverage())
exp.set_round_limit(2)
exp.set_training_data(training_data=None, from_tags=True)
exp.set_job()
exp.set_strategy(node_selection_strategy=DefaultStrategy)

# here we are adding an Optimizer on Researcher side (FedYogi)
fed_opt = Optimizer(lr=.8, modules=[FedYogi()])
exp.set_agg_optimizer(fed_opt)

exp.run(increase=True)
```


!!! warning "Important": ***you may have noticed that we are using `FedAvg` in the `Experiment` configuration, while using `YogiModule` as an `Optimizer`. In fact, `FedAvg` `Aggregator` in `Fed-BioMed` refers to the way model weights are aggregated, and should not be confused with the [whole `FedAvg` algorithm](https://arxiv.org/abs/1602.05629), which is basically a SGD optimizer performed on `Node` side using `FedAvg` `Aggregtor`.**

One can also pass directly the `agg_optimizer` in the `Experiment` object constructor:

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators import FedAverage
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from declearn.optimizer.modules import YogiModule as FedYogi


tags = ['#my-data']
fed_opt = Optimizer(lr=.8, modules=[FedYogi()])


exp = Experiment(tags=tags,
                 training_plan_class=MyTrainingPlan,
                 round_limit=2,
                 agg_optimizer=fed_opt,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)

exp.run(increase=True)

```

## `declearn` auxiliary variables based `Optimizers`

In this subsection, we will see some specific `Optimizers` that are built around `auxiliary variables`.

### What is an auxiliary variable?

`Auxiliary variable` is a parameter that is needed for an `Optimizer` that requieres to be exchanged between `Nodes` and `Researcher`, in addition to model parameters. [`Scaffold`](https://arxiv.org/abs/1910.06378) is an example of such `Optimizer`, because built upon correction states, exchanged from `Nodes` and `Researcher`.

These `Optimizers` may come with a specific `Researcher` version (for `Scaffold` it is `ScaffoldServerModule`) and a `Node` version (resp. `ScaffoldClientModule`). They may work in a synchronous fashion: `Researcher` optimizer version may expect auxiliary variables from  `Node` optimizer, and the other way arround (`Node` optimizer expecting auxiliary variable input from `Reseracher` optimizer version).

### An example using `Optimizer` with auxiliary variables: `Scaffold` with `declearn`

In the last sub-section, we introduced [`Scaffold`](https://arxiv.org/abs/1910.06378). Let's see now how to use it in `Fed-BioMed`.

**`Training Plan` design**

We showcase how to edit your `Training Plan` for PyTorch in order to use `Scaffold`
```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.optimizers.optimizer import Optimizer
from declearn.optimizer.modules import ScaffoldClientModule
...

class MyTrainingPlan(TorchTrainingPlan):
    ...

    def init_dependencies(self):
        deps = [
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from declearn.optimizer.modules import ScaffoldClientModule",
                ]

        return deps
    
    def init_optimizer(self):
        return Optimizer(lr=.01, modules=[ScaffoldClientModule()])

```

**`Experiment` design**


```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators import FedAverage
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from declearn.optimizer.modules import ScaffoldServerModule


tags = ['#my-data']
fed_opt = Optimizer(lr=.8, modules=[ScaffoldServerModule()])


exp = Experiment(tags=tags,
                 training_plan_class=MyTrainingPlan,
                 round_limit=2,
                 agg_optimizer=fed_opt,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)

exp.run(increase=True)

```

!!! warning "Important": ***you may have noticed that we are using `FedAvg` in the `Experiment` configuration, while using `YogiModule` as an `Optimizer`. In fact, `FedAvg` `Aggregator` in `Fed-BioMed` refers to the way model weights are aggregated, and should not be confused with the [whole `FedAvg` algorithm](https://arxiv.org/abs/1602.05629), which is basically a SGD optimizer performed on `Node` side using `FedAvg` `Aggregtor`.**


You can find more examples in [Advanced Optimizers tutorial](../../tutorials/optimizers/01-fedopt-and-scaffold.ipynb)

## Table to use common Federated Learning algorithm with `declearn` in `Fed-BioMed`

## Common Pitfalls using `declearn` Optimizers in `Fed-BioMed`

Below, we are summerizing all common pitfalls that may occur when using `declearn` package in `Fed-BioMed`:
- `Optimization` on `Researcher` side is only possible through `declearn` Optimizers (and not through native Optimizer such as PyTorch Optimizers);
- Some `Optimizers` may requiere some synchronization: it is the case of `ScaffoldClientModule` and `ScaffoldServerModule`;
- For the moment `declearn` Optimizers that use `auxiliary variables` (such as `Scaffold`) is not compatible yet with `SecAgg`.
- check for inconcistent Optimizers! Using a `Regularizer` on `Researcher` side may be non-sensical.
- `Scaffold` aggregator must not be used 

# Conclusion
