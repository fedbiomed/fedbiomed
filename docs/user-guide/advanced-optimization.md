---
title: Advanced optimization in Fed-BioMed with dclearn
description: >-
    Fed-BioMed can interface with declearn to propose powerful and advanced Optimizers
keywords: optimization,optimizer,declearn,OptiModule,Regularizer,regularization
---

# Advanced Optimization in Fed-BioMed

Advanced Optimization can be done in `Fed-BioMed` through the use of `declearn`, a Python package that provides gradient-based `Optimizers`. `declearn` is cross-machine learning framework, meaning that it can be used with most machine learning frameworks (scikit-learn, PyTorch, Tensorflow, JAX, ...).

The following chapter explores in depth how to use `declearn` optimization feature in `Fed-BioMed`. For an example, please refer to the [Advanced Optimizer tutorial](../../tutorials/optimizers/01-fedopt-and-scaffold).

## 1. Introduction to `Declearn` based Optimizer: a cross framework `Optimizer` library

### 1.1. What is `declearn` package?

[`declearn` package](https://gitlab.inria.fr/magnet/declearn/declearn2) is another Federated Learning framework modular and combinable, providing state-of-the-art gradient-based `Optimizer` algorithms. In `Fed-BioMed`, we are only using [its `Optimization` facility](https://gitlab.inria.fr/magnet/declearn/declearn2/-/blob/optimizer-guide/docs/user-guide/optimizer.md), leaving aside all other components of `declearn` that we don't use in `Fed-BioMed`.


**References**: For further details about `declearn`, you may visit:

- [`declearn` repository](https://gitlab.inria.fr/magnet/declearn/declearn2)

- [`declearn` general documentation](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/)

- [`declearn` Optimizers documentation](https://gitlab.inria.fr/magnet/declearn/declearn2/-/blob/optimizer-guide/docs/user-guide/optimizer.md)


### 1.2. `declearn` interface in `Fed-BioMed`: the `Optimizer` object

In `Fed-BioMed`, we provide a `Optimizer` object, that works as an interface with `declearn`, and was made in order to use `declearn`'s Optimizers (see below [`declearn`'s `OptiModules`](#declearn-optimodules) and [`declearn`'s `Regularizers`](#declearn-regularizers)).

```python
from fedbiomed.common.optimizers import Optimizer

Optimizer(lr=.1, decay=.0, modules=[], regualrizers=[])
```

with the following arguments:

 - `lr`: the learning rate;

 - `decay`: the weight decay;

 - `modules`: a list of `declearn` 's `OptiModules` (or a list of `OptiModules' names`);

 - `regularizers`: a list of `declearn` 's `Regularizers` (or a list of `Regularizers' names`).

<a name="declearn-optimodules" ></a>

### 1.3. `declearn`'s `OptiModules`

`declearn` `OptiModules` are modules that convey `Optimizers`, which purpose is to optimize a loss function (that can be written using a PyTorch loss function or defined in a scikit learn model) in order to optimize a model. Compatible `declearn` `OptiModules` with Fed-BioMed framework are defined in `fedbiomed.common.optimizers.declearn` module. They should be imported from `fedbiomed.common.optimizers.declearn`, as shown in the examples below. You can also import them direclty from `declearn`'s `declearn.optimizer.modules`, but they will be no guarentees it is compatible with Fed-BioMed. In fact, recommended method is importing modules through `fedbiomed.common.optimizers.declearn`.

**Usage**:

 - For basic SGD (Stochastic Gradient Descent), we don't need to specify a `declearn` `OptiModule` and/or  a `Regularizer`

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer

    lr = .01
    Optimizer(lr=lr)
    ```

- For a specfic Optimizer like Adam, we need to import `AdamModule` from `declearn`. Hence, it yields:

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from fedbiomed.common.optimizers.declearn import AdamModule

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule()])

    ```

- It is possible to chain `Optimizer` with several `OptiModules`, meaning to use several `Optimizers`. Some chains of `OptiModule` may be non-sensical, so use it at your own risk! Below we showcase the use of Adam with Momentum

    ```python

    from fedbiomed.common.optimizers.optimizer import Optimizer
    from fedbiomed.common.optimizers.declearn import AdamModule, MomentumModule

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule(), MomentumModule()])

    ```

- To get all comptible `OptiModule` in Fed-BioMed, one can run the [`list_optim_modules`][fedbiomed.common.optimizers.declearn.list_optim_modules].

    ```python
    from fedbiomed.common.optimizers.declearn import list_optim_modules

    list_optim_modules()
    ```

For further information on `declearn OptiModule`, please visit [`declearn OptiModule`](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/api-reference/optimizer/Optimizer/) and [`declearn`'s Optimizers documentation](https://gitlab.inria.fr/magnet/declearn/declearn2/-/blob/optimizer-guide/docs/user-guide/optimizer.md).


**List of available Optimizers provided by `declearn`**

To get a list of all available Optimizers in declearn, please enter (after having loaded `Fed-BioMed` conda environment)

```python
from declearn.optimizer import list_optim_modules

list_optim_modules()

```

<a name="declearn-regularizers" ></a>

### 1.4. `declearn`'s `Regularizers`

`declearn`'s `Regularizers` are objects that enable the use of `Regularizer`, which add an additional term to the loss function one wants to Optimize through the use of optimizer. It mainly helps to get a more generalizable model, and prevents overfitting.

The optimization equation yields to:

$$ \theta_{t+1} := \theta_t - \eta  \vec{\nabla} f_{x,y} + \alpha \lVert \theta_t \rVert $$

with

$\theta_t : \textrm{model weights at update } t$

$\eta : \textrm{learning rate}$

$\alpha : \textrm{regularization coefficient}$

$f_{x,y}: \textrm{Loss function used for optimizing the model}$


`Regularizers` should be used and combined with an Optimizer. For instance, SGD with Ridge regression, or Adam with Lasso regression. [`FedProx`](https://arxiv.org/abs/1812.06127) is also considered as a regularization.

!!! note "Optimizer without OptiModules"
    When no `OptiModules` are specified in the `modules` argument of `Optimizer`, plain SGD algorithm is set by default for the optimization.

**Usage**:

- for example, **SGD with Ridge regression** will be written as:

    ```python

    from fedbiomed.common.optimizers.optimizer import Optimizer
    from fedbiomed.common.optimizers.declearn import RidgeRegularizer

    lr = .01
    Optimizer(lr=lr, regularizers=[RidgeRegularizer()])
    ```

- **[Adam](https://arxiv.org/abs/1412.6980) with Lasso Regression**:

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from fedbiomed.common.optimizers.declearn import AdamModule
    from fedbiomed.common.optimizers.declearn import LassoRegularizer

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule()], regularizers=[LassoRegularizer()])

    ```

- Chaining several Regularizers: an example with Ridge and Lasso regularizers, and [Adam](https://arxiv.org/abs/1412.6980) with momentum as the Optimizer.

    ```python
    from fedbiomed.common.optimizers.optimizer import Optimizer
    from fedbiomed.common.optimizers.declearn import LassoRegularizer, RidgeRegularizer

    lr = .01
    Optimizer(lr=lr, modules=[AdamModule(), MomentumModule()], regularizers=[LassoRegularizer(), RidgeRegularizer()])
    ```

For further information on `declearn Regularizer`, please visit [`declearn Regularizers` documentation webpage](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/api-reference/optimizer/regularizers/Regularizer/) 

### 1.5. Chaining `Optimizers` and `Regularizers` with `declearn` modules

It is possible in `declearn` to chain several `OptiModules` and `Regularizers` in an Optimizer.
Generaly speaking, `Optimizer` in `declearn` can be written as:

$$
 \begin{equation}
    \theta_{t+1} := \theta_t - \eta\  \underbrace{ Opt( \vec{\nabla} f_{x,y} - \overbrace{ Reg(\theta_t)}^\textrm{Regularizer})}_\textrm{OptiModule} - \tau \theta_t 
 \end{equation} 
 $$ 

with

$Opt : \textrm{an OptiModule}$

$Reg : \textrm{a Regularizer}$

$\theta : \textrm{model weight}$

$\eta : \textrm{learning rate}$

$\tau : \textrm{weight decay}$

$f_{x,y}: \textrm{Loss function used for optimizing the model}$

The above holds for a single `Regularizer` and `OptiModule`. When using (ie *chaining*) several `OptiModules` and `Regularizers` in an `Optimizer`, the above optimization equation becomes:


$$\theta_{t+1} := \theta_t - \eta\  \underbrace{ Opt_{i=1} \circ Opt_{i=2} \,\circ... \circ \, Opt_{i=n}( \vec{\nabla} f_{x,y} - \overbrace{ Reg_{i=1}\circ Reg_{i=2}\circ... \circ \,Reg_{i=m}(\theta_t)}^\textrm{ Regularizers})}_\textrm{OptiModules} - \tau \theta_t $$


where

$Opt_{1\le i \le n}: \textrm{ OptiModules, with }  n   \textrm{ the total number of OptiModules used}$

$Reg_{1\le i \le m}: \textrm{ Regularizers, with }  m   \textrm{ the total number of Regularizers used}$

**Example**: let's write an `Optimizer` using `RMSProp` and `Momentum` `OptiModules`, and both Lasso and Ridge Regularizers.


```python
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn  import RMSPropModule, MomentumModule, LassoRegularizer, RidgeRegularizer

lr = .01
Optimizer(lr=lr,
          modules=[RMSPropModule(), MomentumModule()],
          regularizers=[LassoRegularizer(), RidgeRegularizer()])
```

!!! note "Using list of strings instead of list of modules"
    In `declearn`, it is possible to use name of modules instead of loading the actual modules. In the script below, we are rewritting the same `Optimizer` as the one above but by specifying the module names. A convinient way to get the naming is to use `list_optim_modules` and  `list_optim_regularizers` functions, that map module names with their classes respectively.

```python
from fedbiomed.common.optimizers.optimizer import Optimizer


lr = .01
Optimizer(lr=lr,
          modules=['adam', 'momentum'],
          regularizers=['lasso', 'ridge'])

```

To get to know specifcities about all `declearn`'s modules, please visit [`declearn` webpage](https://magnet.gitlabpages.inria.fr/declearn/docs/2.2/).

#### How to use well-known Federated-Learning algorithms with `declearn` in `Fed-BioMed`?

Please refer to [the following section of this page.](#federated-learning-algorithms-table)

## 2. `declearn` optimizer on Node side

In order to use `declearn` to optimize `Node`s local model, you will have to edit `init_otpimizer` method in the `TrainingPlan`. Below we showcase how to use it with PyTorch framework (using Adam and Ridge regularizer for the optimization).

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn  import AdamModule, RidgeRegularizer
...

class MyTrainingPlan(TorchTrainingPlan):
    ...

    def init_dependencies(self):
        deps = [
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from fedbiomed.common.optimizers.declearn  import AdamModule, RidgeRegularizer"
                ]

        return deps
    
    def init_optimizer(self):
        return Optimizer(lr=.01, modules=[AdamModule()], regularizers=[RidgeRegularizer()])


```

!!! note "Important"
    You should specify the `OptiModules` imported in both the imports at the begining of the `Training Plan` as well as in the dependencies (in the `init_dependencies` method within the `Training Plan`). The same holds for `declearn`'s `Regularizers`.

Syntax will be the same for scikit-learn as shown below, using the same `Optimizer`:

```python
from fedbiomed.common.training_plans import FedSGDClassifier
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn import AdamModule, RidgeRegularizer
...

class MyTrainingPlan(FedSGDClassifier):
    ...

    def init_dependencies(self):
        deps = [
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from fedbiomed.common.optimizers.declearn  import AdamModule, RidgeRegularizer"
                ]

        return deps
    
    def init_optimizer(self):
        return Optimizer(lr=.01, modules=[AdamModule()], regularizers=[RidgeRegularizer()])

```

## 3. `declearn` optimizer on Researcher side (`FedOpt`)

`Fed-BioMed` provides a way to use  **Adaptive Federated Optimization**, introduced as [`FedOpt` in this paper](https://arxiv.org/pdf/2003.00295.pdf). In the paper, authors considered the difference of the global model weights between 2 successive `Rounds` as a *pseudo gradient*, paving the way to the possbility to have `Optimizers` on `Researcher` side, optimizing the updates of the global model.
To do so, `fedbiomed.researcher.experiment.Experiment` has a method to set the `Researcher Optimizer`: [`Experiment.set_agg_optimizer`](../../developer/api/researcher/experiment/#fedbiomed.researcher.experiment.Experiment.set_agg_optimizer)

Below an example using the `set_agg_optimizer` with `FedYogi`:

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators import FedAverage
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.common.optimizers.declearn  import YogiModule as FedYogi

tags = ['#my-data']

exp = Experiment()
exp.set_training_plan_class(training_plan_class=MyTrainingPlan)
exp.set_tags(tags = tags)
exp.set_aggregator(aggregator=FedAverage())
exp.set_round_limit(2)
exp.set_training_data(training_data=None, from_tags=True)
exp.set_strategy(node_selection_strategy=DefaultStrategy)

# here we are adding an Optimizer on Researcher side (FedYogi)
fed_opt = Optimizer(lr=.8, modules=[FedYogi()])
exp.set_agg_optimizer(fed_opt)

exp.run(increase=True)
```

!!! warning "Important"
    **You may have noticed that we are using `FedAverage` in the `Experiment` configuration, while using `YogiModule` as an `Optimizer`. In fact, `FedAverage` `Aggregator` in `Fed-BioMed` refers to the way model weights are aggregated before optimization, and should not be confused with the [whole `FedAvg` algorithm](https://arxiv.org/abs/1602.05629), which is basically a SGD optimizer performed on `Node` side using `FedAvg` `Aggregtor` on `Researcher` side.**

One can also pass directly the `agg_optimizer` in the `Experiment` object constructor:

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators import FedAverage
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.common.optimizers.declearn import YogiModule as FedYogi


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

## 4. `declearn` auxiliary variables based `Optimizers`

In this subsection, we will take a look at some specific `Optimizers` that are built around `auxiliary variables`.

### 4.1. What is an auxiliary variable?

`Auxiliary variable` is a parameter that is needed for an `Optimizer` that requieres to be exchanged between `Nodes` and `Researcher`, in addition to model parameters. [`Scaffold`](https://arxiv.org/abs/1910.06378) is an example of such `Optimizer`, because built upon correction states, exchanged from `Nodes` and `Researcher`.

These `Optimizers` may come with a specific `Researcher` version (for `Scaffold` it is `ScaffoldServerModule`) and a `Node` version (resp. `ScaffoldClientModule`). They may work in a synchronous fashion: `Researcher` optimizer version may expect auxiliary variables from  `Node` optimizer, and the other way arround (`Node` optimizer expecting auxiliary variable input from `Reseracher` optimizer version).

### 4.2. An example using `Optimizer` with auxiliary variables: `Scaffold` with `declearn`

In the last sub-section, we introduced [`Scaffold`](https://arxiv.org/abs/1910.06378). Let's see now how to use it in `Fed-BioMed` framework.


!!! note "About native Scaffold implementation in Fed-BioMed"
    `Fed-BioMed` provides its own implementation of [`Scaffold` `Aggregator`](https://fedbiomed.org/latest/developer/api/researcher/aggregators/#fedbiomed.researcher.aggregators.Scaffold), that is only for PyTorch framework. It only works with PyTorch native optimizers `torch.optim.Optimizer` for the `Node Optimizer`.


**`Training Plan` design**

In the current subsection, we showcase how to edit your `Training Plan` for PyTorch in order to use `Scaffold`

```python
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn import ScaffoldClientModule
...

class MyTrainingPlan(TorchTrainingPlan):
    ...

    def init_dependencies(self):
        deps = [
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from fedbiomed.common.optimizers.declearn import ScaffoldClientModule",
                ]

        return deps
    
    def init_optimizer(self):
        return Optimizer(lr=.01, modules=[ScaffoldClientModule()])

```

**`Experiment` design**

This is how `Experiment` can be designed (on the `Researcher` side)

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

!!! warning "Important"
    **You may have noticed that we are using `FedAverage` in the `Experiment` configuration, while using `ScaffoldServerModule` \ `ScaffoldClientModule` as an `Optimizer`. In fact, `FedAverage` `Aggregator` in `Fed-BioMed` refers to the way model weights are aggregated before optimization, and should not be confused with the [whole `FedAvg` algorithm](https://arxiv.org/abs/1602.05629), which is basically a SGD optimizer performed on `Node` side using `FedAvg` `Aggregtor`.**


!!! warning "Security issues using auxiliary variables when using SecAgg"
    Currently, `declearn` optimizers based on auxiliary variables (like `Scaffold`), do not have their auxiliary variables protected by [`SecAgg`](../../user-guide/secagg/introduction) secure aggregation mechanism yet. This is something that will be changed in future `Fed-BioMed` releases. 

You can find more examples in [Advanced Optimizers tutorial](../../tutorials/optimizers/01-fedopt-and-scaffold.ipynb)

## Table to use common Federated Learning algorithm with `declearn` in `Fed-BioMed`

<a name="federated-learning-algorithms-table" ></a>

Below we have gathered some of the most well known algorithms in Federated Learning in the following table (as a reminder, `Node Optimizer` must e defined in the `TrainingPlan`, whereas `Researcher Optimizer` in the `Experiment` object):

| Federated Learning Algorithm                                       	| Node Optimizer                                             	| Researcher Optimizer                                     	| Aggregator   	|
|--------------------------------------------------------------------	|------------------------------------------------------------	|----------------------------------------------------------	|--------------	|
| [AdaAlter (distributed AdaGrad)](https://arxiv.org/pdf/1911.09030) 	| AdaGrad  `Optimizer(lr=xx, modules=[AdaGradModule()])`     	| None                                                     	| `FedAverage` 	|
| [FedAdagrad](https://arxiv.org/pdf/2003.00295)                     	| SGD `Optimizer(lr=xx)`                                     	| AdaGrad `Optimizer(lr=xx, modules=[AdaGradModule()])`    	| `FedAverage` 	|
| [FedAdam](https://arxiv.org/pdf/2003.00295)                        	| SGD `Optimizer(lr=xx)`                                     	| Adam `Optimizer(lr=xx, modules=[AdamModule()])`          	| `FedAverage` 	|
| [FedAvg](https://arxiv.org/abs/1602.05629)                         	| SGD `Optimizer(lr=xx)`                                     	| None                                                     	| `FedAverage` 	|
| [FedProx](https://arxiv.org/abs/1812.06127)                        	| SGD  `Optimizer(lr=xx, regularizers=[FedProxRegularizer])` 	| None                                                     	| `FedAverage` 	|
| [FedYogi](https://arxiv.org/pdf/2003.00295)                        	| SGD `Optimizer(lr=xx)`                                     	| Yogi `Optimizer(lr=xx, modules=[YogiModule()])`          	| `FedAverage` 	|
| [Scaffold](https://arxiv.org/abs/1910.06378)                       	| SGD `Optimizer(lr=xx, modules=[ScaffoldClientModule()])`   	| SGD `Optimizer(lr=xx, modules=[ScaffoldServerModule()])` 	| `FedAverage` 	|


## 5. Common Pitfalls using `declearn` Optimizers in `Fed-BioMed`

Below, we are summerizing common pitfalls that may occur when using `declearn` package in `Fed-BioMed`:

- `Optimization` on `Researcher` side is only possible through `declearn` Optimizers (and not through native Optimizers such as PyTorch Optimizers);
- Some `Optimizers` may requiere some synchronization: it is the case of `Scaffold` related modules, ie `ScaffoldClientModule` and `ScaffoldServerModule`;
- For the moment `declearn` Optimizers that use `auxiliary variables` (such as `Scaffold`) cannot be protected yet with [Secure Aggregation](../../user-guide/secagg/introduction/);
- For the moment, `declearn`'s `optimizer` only comes with a unique learning rate (multiple learning rates `Optimizers`, for example pytorch optimizers `torch.optim.Optimizer` can handle a learning rate per model layer );
- When chaining `declearn`'s `OptiModules`, it is only possible to use a unique learning rate, that will be the same for all `OptiModules`, and that won't change during a `Round`;
- check for inconcistent Optimizers! Using a `Regularizer` on `Researcher` side may be non-sensical, even if it is doable within `declearn`;
- [`Scaffold` `Fed-BioMed` aggregator](https://fedbiomed.org/latest/developer/api/researcher/aggregators/#fedbiomed.researcher.aggregators.Scaffold)  must not be used when using both `ScaffoldServerModule` and `ScaffoldClientModule`. This `aggregator` is in fact an alternative to the `declearn` `scaffold`, and you have to choose between the `Fed-BioMed` native version of `Scaffold` and the `declearn` 's one. Please note that `Fed-BioMed aggregator Scaffold` is deprecated, hence, the use of `ScaffoldServerModule` and `ScaffoldClientModule` is highly encouraged.

## Conclusion

We have seen how to use `declearn` `Optimizers` in `Fed-BioMed`. In `Fed-BioMed`, it is possible to set an `Optimizer` on both the `Node` and the `Researcher` side:

- On `Node` side, such `Optimizer` is defined in `Training Plan` and is used to optimize `Nodes`' local models;

- On `Researcher` side, `Optimizer` is defined in the `Experiment`, and is made for optimizing global model.


When used with `declearn` package, `Fedd-BioMed` `Aggregator` is used for aggregating weights, before any potential optiization: `FedAverage` does the weighted sum of all local models sent back by the `Nodes`.

`declearn` comes with the possibility of *chaining* `Optimizers`, by passing a list of `OptiModule` and `Regularizers`, making possible to try out some more complex optimization process.

Check [the tutorial related to the use of `declearn`'s `Optimizers`](../../tutorials/optimizers/01-fedopt-and-scaffold.ipynb)
