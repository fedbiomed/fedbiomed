# Federated pre-processing

## Overview

Federated pre-processing applies data transformations across nodes without moving raw data.
The result is a new federated dataset, updated node by node.

At the moment, Fed-BioMed provides one federated pre-processing method: **FedComBat**.

## FedComBat in short

FedComBat harmonizes tabular data to reduce site effects.
In this context, one site corresponds to one Fed-BioMed node.

- **Phenotypes**: modified variables to harmonize.
- **Covariates**: predictive variables whose effects are modeled and preserved.

Reference paper: [Federated ComBat](https://arxiv.org/pdf/2601.14314).

## Requirements

- Datasets must be in CSV format.
- At least 2 nodes must participate.
- Node-side federated pre-processing must be enabled.
- Node-side federated analytics must be enabled (FedComBat uses federated statistics).

See node configuration details in [Configuring Nodes](../nodes/configuring-nodes.md) and [Federated Analytics - Nodes](../nodes/federated-analytics.md).

## Configure FedComBat in an experiment

Attach pre-processing to an `Experiment` with `set_preprocessing`:

```python
from fedbiomed.common.constants import PreprocType

fedcombat_args = {
	"covariates": ["SEX", "AGE", "PTEDUCAT"],
	"phenotypes": ["CDRSB.bl", "ADAS11.bl"],
}

exp.set_preprocessing(PreprocType.FEDCOMBAT, fedcombat_args)
```

### Arguments

Mandatory:

- `covariates`: list of column names or list of column indices.
- `phenotypes`: list of column names or list of column indices.

Constraints:

- `covariates` and `phenotypes` must be lists.
- They must use the same type (`str` or `int`).
- They must be disjoint (no duplicate across both lists).

Optional:

- `rounds`: FL rounds for the internal FedComBat model training.
- `training_args`: training arguments for internal FedComBat model training.
- `model_args`: additional model arguments for internal FedComBat model training.
- `standardize_result` (`False` by default): if `False` restore initial value scale for harmonized output. If `True`, keep harmonized output in standardized scale.

## Execution behavior

When you run training (`exp.run()` or `exp.run_once()`), pre-processing is executed automatically if needed.

So after attaching a pre-processing to an experiment, next `exp.run()` ensures the experiment dataset is harmonized across the nodes before the experiment model is trained. This means the experiment trains on the (automatically) harmonized dataset.

You can also trigger harmonization manually:

```python
exp.preprocessing.execute()  # executes only if needed
exp.preprocessing.execute(force=True)  # force re-run even if not needed
```

FedComBat is re-executed only when context changes, for example:

- selected nodes changed,
- federated dataset changed,
- pre-processing configuration changed.

After harmonization, the experiment training data is updated in place:

```python
exp.training_data().data()
```

!!! warning "Important"
    If pre-processing is re-executed, it is re-executed depending on the content of the federated dataset.
    For example, if one node was removed from the federated dataset, it can lead to re-harmonize already harmonized
    dataset. This may not be the desired behaviour. In this case, one may want to re-execute harmonization
    on the original datasets of the remaining nodes.



## Training plan approval (secure setups)

In deployments requiring training plan approval, FedComBat's internal training plan must be approved before execution.

Submit FedComBat's internal training plan to the nodes for approval, then wait for the node to approve it 
before running the pre-processing:

```python
exp.preprocessing.approve("FedComBat harmonization models")
```

See [Training with approved training plans](../../tutorials/security/training-with-approved-training-plans.ipynb).

## VPN / container mode

In VPN/container deployments, pre-processing and federated analytics are disabled by default.
Enable both features on node containers to be able to successfully run FedComBat pre-processing.

```bash
export FBM_SECURITY_ALLOW_PREPROC=True
export FBM_SECURITY_ALLOW_FEDERATED_ANALYTICS=True
```

See [Deployment in VPN mode](../deployment/deployment-vpn.md).

