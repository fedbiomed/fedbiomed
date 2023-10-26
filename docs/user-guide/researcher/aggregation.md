---
title: Parameter Aggregation in Fed-BioMed
description: Aggregation of model parameters in Fed-BioMed for dealing with data heterogeneity.
keywords: parameter aggregation,aggregation,federated average
---


# Parameter Aggregation in Fed-BioMed

Aggregation of model parameters plays an important role in federated learning, where we naturally deal with data 
heterogeneity. Unlike the distributed learning datasets, model parameters are saved as same-sized data blocks for 
each node training the same model. The number of samples, the quality of the samples cand their data distribution can vary in every `Node`. 
In Fed-BioMed, we currently work on providing various solutions for this heterogeneity. Up to now, we support 
[`FedAverage`](https://arxiv.org/abs/1602.05629) which performs the standard aggregation scheme in federated learning: federated averaging. We also provide [`FedProx`](https://arxiv.org/abs/1812.06127) and [`SCAFFOLD`](https://arxiv.org/pdf/2207.06343.pdf) aggregation methods.

!!! warning "Important"
    The following `Aggregators` can also be used with `declearn Optimizers`, providing advanced gradient-based optimization modules.
    These `Optimizers` are cross-framework, meaning it is possible to use it with all machine learning framework provided by Fed-BioMed. Please visit the webpage dedicated to [*advanced optimization*](../../advanced-optimization).
 
## Fed-BioMed `Aggregators`:

Fed-BioMed `Aggregators` are showcased in the following [tutorial](../tutorials/pytorch/05-aggregation-in-fed-biomed). 

### Federated Averaging (FedAveraging)

`FedAveraging` is the default `Aggregator` in Fed-BioMed, introduced by [McMahan et al.](https://arxiv.org/abs/1602.05629). It 
performs a weighted mean of local model parameters based on the size of node specific datasets. This operation 
occurs after each round of training in the `Nodes`.

$$w_{t+1} := \sum_{k=1}^K\frac{n_k}{n}w_{t+1}^k$$

where \( w_{t} \) are the weights at round $t$, $K$ is the number of `Nodes` participating at round $t$, and \( n_k, n \) 
are the number of samples of the \(k\)-th node and of the total federation respectively. 

### FedProx

Similar to `FedAveraging`, [`FedProx`](https://arxiv.org/abs/1812.06127) performs a weighted sum of local model parameters. 
`FedProx` however introduces a regularization operation, using $\mathcal{L}_2$ norm, in order to tackle statistical heterogeneity. 
Basically, it reformulates the loss function by:

$$F_k(w) + \frac{\mu}{2}|| w - w^t ||^2_2$$ 

using the same notation as above, with $\mu$ the regularization parameter (we obtain `FedAveraging` by setting $\mu=0$) and $F_k$ the objective function.

To use `FedProx`, use `FedAverage` from `fedbiomed.researcher.aggregators` and specify a value for $\mu$ in the training 
arguments `training_args` using the argument name `fedprox_mu`.


### SCAFFOLD

[`SCAFFOLD`](https://arxiv.org/pdf/2207.06343.pdf) stands for *Stochastic Controlled Averaging for Federated Learning*. 
It introduces a correction state parameter in order to tackle *the client drift*, depicting the fact that when data 
across each `Node` are heterogeneous, each of the `Nodes` pushes the model in a different direction in the optimization 
space and the global model does not converge towards the true optima. 
In Fed-BioMed, only *option 2* of the `SCAFFOLD` paper has been implemented. 
Additional details about the implementation can be found in the developer 
[API reference][fedbiomed.researcher.aggregators.Scaffold].

The corrected loss function used to update the model is computed as follows:

$$F_k(w) + c \cdot w - c_k \cdot w$$

where $c_k$ is the `Node` correction term,  $c = \frac{1}{K}\sum_{k=1}^K{c_k}$ is the server's correction term,  
and $K$ is the total number of participating `Nodes` as above. 

On the `Researcher` side, the global model is updated by performing gradient descent.

Additional parameters are needed when working with `SCAFFOLD`: 

 - `server_lr`: `Researcher`'s learning rate for performing a gradient step
 - `num_updates`: the number of updates (ie gradient descent optimizer steps) to be performed on each `Node`. Relying only on `epochs` could lead to some inconsistencies in the computation of the correction term: thus, in Fed-BioMed, `SCAFFOLD` aggregator cannot be used with `epochs`.

Please note that:

 - `SCAFFOLD` should be used only with `SGD` optimizer. Using other `Optimizers` in Fed-BioMed is possible, but without any convergence guarantees.
 - `SCAFFOLD` can only be used with the `PyTorch` framework at the moment.
 - `SCAFFOLD` **requires** using the `num_updates` training argument to control the number of [training iterations](./experiment.md#controlling-the-number-of-training-loop-iterations). Using only `epochs` will raise an error.


## How to Create Your Custom Aggregator

### Designing your own `Aggregator` class: the `aggregation` method

Before designing your custom aggregation algorithm we recommend you to see default `FedAverage` 
[aggregate method][fedbiomed.researcher.aggregators.fedavg.FedAverage.aggregate]


`aggregate` method is expecting at least `model_params` and `weights` arguments. Additional argument can be passed 
through `*args` and `kwargs` depending, on the values needed for your `Aggregator`.

It is possible to create your custom aggregator by creating a new class which inherits from the Aggregator class 
defined in `fedbiomed.researcher.aggregators.aggregator.Aggregator`.

```python
class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        pass

    @staticmethod
    def normalize_weights(weights) -> list:
        # Load list of weights assigned to each node and
        # normalize these weights so they sum up to 1
        norm = [w/sum(weights) for w in weights]
        return norm

    def aggregate(self,  model_params: list, weights: list, *args, **kwargs) -> Dict: # pragma: no cover
        """Strategy to aggregate models"""
        pass

```

Your child class should extend the method `aggregate` that gets model parameters and weights as arguments. The model 
parameters are those which have been locally updated in each node during the last round. The weights represent the 
ratio of the number of samples in each node and the total number of samples. Your custom aggregator class should return 
aggregated parameters.

You should also pay attention to the way the parameters are loaded. For example, it may be a dictionary that contains 
tensor data types or just an array. As you can see from the following example, the aggregator first checks the data 
type of the parameters, and then it does the averaging.

```python
    if t == 'tensor':
        for model, weight in zip(model_params, proportions):
            for key in avg_params.keys():
                avg_params[key] += weight * model[key]

    if t == 'array':
        for key in avg_params.keys():
            matr = np.array([ d[key] for d in model_params ])
            avg_params[key] = np.average(matr,weights=np.array(weights),axis=0)
```

### Desinging your own `Aggregator` class: the `create_aggregator_args` method

For some advanced `Aggregators`, you may need to send some argument to `Nodes` in order to update the local model. For instance,
`SCAFFOLD` `Aggregator` sends specific correction terms for each of the `Nodes` involved in the training. 

The method that has this responsability is `create_aggregator_args`, and is designed as follow (in the `fedbiomed.researcher.aggregators.aggregator.Aggregator` class):

```python
def create_aggregator_args(self, *args, **kwargs) -> Tuple[dict, dict]:
    """Returns aggregator arguments that are expecting by the nodes
    
    Returns:
    dict: contains `Aggregator` parameters that will be sent to nodes 
    dict: contains parameters that will be sent through file exchange message.
            Both dictionaries are mapping node_id to `Aggregator` parameters specific 
            to each Node.

    """
    return self._aggregator_args or {}, {}
```


## Conclusions

In this article, the aggregation process has been explained. Currently, Fed-BioMed only supports the vanilla federated 
averaging scheme for the aggregation operation called `FedAverage`, as well as `FedProx` and `SCAFFOLD`. However, Fed-BioMed also allows you to create 
your custom aggregator using the `Aggregator` parent class. It means that you can define your custom aggregator based 
on your use case(s). You can define it in your notebook or python script and passed into the 
[Experiment](./experiment.md) as an argument.
