# Using GPU accelerator hardware

Federated learning often implies intensive computation for model training, aggregation or inference.
Dedicated accelerator hardware such as Nvidia GPUs can help speed up these steps, with support from the libraries and frameworks.

This page explains GPU support in Fed-BioMed.


## Support scope

Fed-BioMed supports accelerator hardware with the following requirements and limitations :

* **Nvidia GPU** hardware : basically all models supported by the CUDA interface can be used, but of course the hardware needs to have enough memory for the targeted model.
* **single GPU** on each node : if the host machine has multiple GPUs, only one is used by a Fed-BioMed node
* **PyTorch** framework using the `TorchTrainingPlan` Fed-BioMed training plan interface ; other frameworks / training plans (scikit-learn / `SGDSkLearnModel`) can run on a node with a hardware accelerator but don't use it.
* **only training** is accelerated (node side computation), not aggregation (server side computation)
* **no VPN** : VPN in Fed-BioMed is based on running each component in a docker container. Node container images provided by Fed-BioMed are not yet GPU-enabled.


## Node side

To control GPU usage by Fed-BioMed training from the node side, use these options of the
`fedbiomed_run node` command :

* `--gpu` : Node offers to use a GPU for training, if a GPU is available on the node and if the researcher requests use of a GPU. If no GPU is available, or the training plan is not supported for GPU (scikit-learn), or the researcher does not request use of a GPU, then training occurs in CPU.
* `--gpu-num GPU_NUM` : Node chooses the device with number *GPU_NUM* in CUDA instead of the default device. If this device does not exist, node fallbacks to default CUDA device. This option also implicitely sets `--gpu`. 
* `--gpu-only` : Node enforces use of a GPU for training, if a GPU is available on the node, even if the researcher doesn't request it. If no GPU is available, or the training plan is not supported for GPU (scikit-learn), then training occurs in CPU.

**By default (no options), Fed-BioMed training doesn't use GPU.**

The reason for not using GPU by default is that even if you have a GPU on a node, it may not have enough memory to train the given model. In this case, the training of a correct model fails with an error message (and you don't want a correct model to fail with default options) :

```shell
2022-01-13 08:07:28,737 fedbiomed ERROR - Cannot train model in round: CUDA error: out of memory
```


Example :

* launch a node that enforces use of GPU with CUDA device number 2 (the 3rd GPU on this host machine). If there is no GPU with device number 2, use the default GPU. If there is no GPU available or if not using `TorchTrainingPlan`, do the training in CPU :
```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start --gpu-only --gpu-num 2
```
* if the researcher didn't request for GPU usage, and there is no GPU numbered 2 in CUDA on the node but another GPU is available, and the training plan supports GPU acceleration, then the following warning messages are emitted :
```shell
2022-01-21 12:34:31,992 fedbiomed WARNING - Node enforces model training on GPU, though it is not requested by researcher
2022-01-21 12:34:31,992 fedbiomed WARNING - Bad GPU number 2, using default GPU
```


## Researcher side

To control GPU usage for Fed-BioMed training from the researcher side, set the `'use_gpu': True` key of the `training_args` dict passed as argument to `Experiment`.

In this example, the researcher requests the nodes participating in the `Experiment` to use GPU for training, if they have any GPU available and offer to use it :
```shell
# Researcher notebook or script code
training_args = {
    'use_gpu': True 
    #......
}

exp = Experiment(
    ...
    training_args=training_args,
    ... )
```

**Node requirements have precedence over researcher requests.** For example, no GPU is used if the node requests it but the nodes does not offer it.


## How to enable GPU usage

Fed-BioMed offers a simplified interface for training with GPU, as described above. This hides from the researcher the complexity of the specific resources and requirements of each node's. 

!!! warning "Warning"
        **Fed-BioMed models and training plans should never try to directly access the CUDA resources on the node.**
        For example don't use the `pytorch.cuda.is_available()`, `tensor.to()`, `tensor.cuda()` etc. methods. This is not 
        the supported way of using GPUs in Fed-BioMed.

### Option 1 : enable GPU on node and researcher

In this scenario, the node proposes the use of a GPU and the researcher requests the use of a GPU :

* launch node offering GPU usage
```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start --gpu
```

* on the researcher, set the `training_args` of the notebook or script used for training
```shell
# Researcher notebook or script code
training_args = {
    'use_gpu': True
    # .....
}

exp = Experiment(
    ...
    training_args=training_args,
    ... )
exp.run()
```

### Option 2 : force GPU usage on node

In this scenario, no action is needed on the researcher side, no code modification is needed. The node enforces use of GPU :

* launch node enforcing GPU usage
```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start --gpu-only
```

* on the researcher, launch same notebook or script as when using CPU
```shell
# Unmodified researcher notebook or script code
exp = Experiment( ... )
exp.run()
```


## Remarks

### Heterogeneous nodes

When using multiple nodes, they can have different GPU support and requirements. For example an `Experiment` can use 4 nodes with :

* 1 node has no GPU available
* 1 node has GPU available but does not offer GPU
* 1 node has GPU available and offers GPU
* 1 node has GPU available and enforces GPU usage
* etc.

### Multiples nodes on the same host

When running multiples nodes on a same host machine that has multiple GPUs available, each node can use a different GPU. For example on a node with 2 GPUs numbered 0 and 1 :
```shell
# Launch node (in background) offering use of GPU 0
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start --gpu-num 0 &
# Launch node offering use of GPU 1
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start --gpu-num 1 
```

### Security

!!! warning "warning"
        Warning: from a security perspective, a malicious researcher can write a training plan that directly accesses 
        the node's GPU (eg: with `tensor.to()`) even if not offered by the node. This should be addressed 
        by using training plan approval and conducting proper training plan review.
    