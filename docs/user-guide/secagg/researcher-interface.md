# Managing Secure Aggregation on Researcher Side

Researcher component is responsible for managing secure aggregation context setup that prepares necessary elements to apply secure aggregation over encrypted model parameters. Some nodes might require secure aggregation while some of them don't, and some others don't support secure aggregation. Therefore, end-user (researcher) should activate secure aggregation depending on all participating nodes configuration.


## Managing secure aggregation through Experiment

### Activation

By default, secure aggregation is deactivated in [`Experiment`][fedbiomed.researcher.federated_workflows.Experiment] class. It can
be activated by setting the `secagg` as `True`, and the default secure aggregation scheme is [LOM](./introduction.md#low-overhead-masking-lom).

```python
from fedbiomed.researcher.federated_workflows import Experiment
Experiment(
    secagg=True
)
```

Setting secagg `True` instantiates a [`SecureAggregation`][fedbiomed.researcher.secagg.SecureAggregation]
with default arguments as [`timeout`](#timeout) and [`clipping_range`](#clipping-range).  However, it is also possible
to create a secure aggregation instance by providing desired argument values.

```python
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.secagg import SecureAggregation
Experiment(
    #...
    secagg=SecureAggregation(clipping_range=30),
    #....
)
```

The argument `scheme` of [`SecureAggregation`][fedbiomed.researcher.secagg.SecureAggregation] allows to select secure aggregation scheme that is going to be used. However, schemes may require different pre or post configuration on the node side and researcher side. Therefore,  please carefully read the [configuration](./configuration.md) guide before changing secure aggregation scheme.

```python
from fedbiomed.researcher.secagg import SecureAggregation, SecureAggregationSchemes

exp = Experiment(tags=tags,
                 model_args=model_args,
                 training_plan_class=MyTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None,
                 secagg=SecureAggregation(scheme=SecureAggregationSchemes.JOYE_LIBERT),
                 # or custom SecureAggregation(active=<bool>, clipping_range=<int>)
                 save_breakpoints=True)

```

### Timeout

Secure aggregation setup starts specific processing in each Fed-BioMed component that participates in the federated training.
However, these processes and communication delay might be longer or shorter than expected depending on number of
nodes and communication bandwidth. Default timeouts cannot currently be configured through the user API, it is needed to edit the `researcher.secagg.SecaggContext` in the library for each component accordingly.

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

### Model encryption takes too much time

The time of encryption depends on model size. If the model is larger, it is normal that the encryption
takes longer.

### I want to set secure aggregation context without re-running a round.

It is possible to access the secagg instance through the experiment object in order to reset the secure aggregation context by providing a list of parties and the experiment `experiment_id`. This step works for all secure aggregation schemes.

```python
from fedbiomed.researcher.federated_workflows import Experiment

exp = Experiment(secagg=True,
                 #....
                 )

exp.secagg.setup(
    parties= parties=[exp.researcher_id] + exp.filtered_federation_nodes(),
    experiment_id=exp.id,
    researcher_id=exp.researcher_id
)

```
If a context has already been set, you can use the force argument to forcefully recreate the context.
```python
exp.secagg.setup(
    parties= parties=[exp.researcher_id] + exp.filtered_federation_nodes(),
    experiment_id=exp.id,
    researcher_id=exp.researcher_id # or config.get('default', 'id')
    force=True
)
```

The outcome of the setup action can vary depending on the secure aggregation scheme used. For example, in the Joye-Libert scheme, the setup action generates `servkey`, and attaches a default biprime number into its context. In contrast, the LOM scheme only tracks the secure aggregation setup status of the participating nodes. This ensures that all participating nodes have created their own context/elements for training before the system sends the train request.
