# Managing Secure Aggregation on Researcher Side

Researcher component is responsible for managing secure aggregation context setup that prepares necessary
elements to apply secure aggregation over encrypted model parameters. Some nodes might require secure aggregation
while some of them don't, and some others don't support secure aggregation. Therefore, end-user (researcher) should activate secure aggregation depending on all participating nodes configuration.


## Managing secure aggregation through Experiment

### Activation

By default, secure aggregation is deactivated in [`Experiment`][fedbiomed.researcher.experiment.Experiment] class. It can
be activated by setting the `secagg` as `True`.

```python
from fedbiomed.researcher.experiment import Experiment
Experiment(
    secagg=True
)
```

Setting secagg `True` instantiates a [`SecureAggregation`][fedbiomed.researcher.secagg.SecureAggregation]
with default arguments as [`timeout`](#timeout) and [`clipping_range`](#clipping-range).  However, it is also possible
to create a secure aggregation instance by providing desired argument values.

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.secagg import SecureAggregation
Experiment(
    #...
    secagg=SecureAggregation(clipping_range=30),
    #....
)
```

!!! warning "Federated averaging"
    Once the secure aggregation is activated, experiment doesn't use the `aggregator` parameter of the `Experiment` (eg `FedAverage`) for aggregation.
    Secure aggregation aggregates model parameters with its own federated average, but without weighting them.
    Therefore, using `num_updates` instead of
    `epochs` in [`training_args`](../researcher/experiment.md#controlling-the-number-of-training-loop-iterations) is strongly recommended for secure aggregation.


### Timeout

Secure aggregation setup launches MP-SPDZ process in each Fed-BioMed component that participates in the federated training.
However, these processes and communication delay might be longer or shorter than expected depending on number of
nodes and communication bandwidth. The argument `timeout` allows increasing or decreasing the timeout for secure
aggregation context setup.

### Clipping Range

Encryption on the node-side is performed after the quantization of model weights/parameters. However, the maximum
and minimum values of model parameters may vary depending on the technique used. Therefore, the clipping range of
quantization depends on the model, data, or technique. The clipping range should always be greater than or equal to
the maximum model weight value, but kept reasonably low.

By default, the clipping range is set to 3. If the clipping range is exceeded while encrypting model parameters,
a warning is raised instead of failing. Therefore, the end-user is aware that the clipping range should
be increased for the next rounds.


!!! note "Setting clipping range"
    The optimal clipping range depends on the specific scenario and the models being used. In some cases, using too
    high of a clipping range can result in a loss of information and lead to decreased performance. Therefore, it is
    important to carefully choose the appropriate clipping range based on the specific situation and the characteristics
    of the models being used.



## Troubleshooting

### Can not set secure aggregation context on the researcher side

This may be because of the timeout on the researcher side. If you have low bandwidth, connection latency or
many nodes, please try to increase timeout.

### Context is set on the nodes but not on the researcher

This is also because of the timeout issue. It happens when MP-SPDZ completes multi-party computation but
can not send success status back to researcher in time. Therefore, researcher assumes that the secure aggregation
is context is not set properly. Please increase secure aggregation timeout and re-run training round.

### Model encryption takes too much time

The time of encryption depends on model size. If the model is larger, it is normal that the encryption
takes longer.

### I want to set secure aggregation context without re-running a round.

It is possible to access the secagg instance through the experiment object in order to reset the secure
aggregation context by providing a list of parties and the experiment `experiment_id`.

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.environ import environ

exp = Experiment(secagg=True,
                 #....
                 )

exp.secagg.setup(
    parties= parties=[environ["ID"]] + exp.filtered_federation_nodes(),
    experiment_id=exp.id
)

```
If a context has already been set, you can use the force argument to forcefully recreate the context.
```python
exp.secagg.setup(
    parties= parties=[environ["ID"]] + exp.filtered_federation_nodes(),
    experiment_id=exp.id,
    force=True
)
```