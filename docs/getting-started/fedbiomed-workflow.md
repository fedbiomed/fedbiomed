---
title: Federated Training Workflow
description: This page presents the workflow of Fed-BioMed.
keywords: Collaborative Learning, Federated Learning, Federated Analytics, workflow
---

# Fed-BioMed Workflow

We present in the following page a short step-by-step illustration detailing the workflow of Fed-BioMed.

The steps are:

 1. <a class="scroller-link" href="#step-1-setting-up-the-nodes">Setting up the <code>Nodes</code></a>
 2. <a class="scroller-link" href="#step-2-deploying-datasets-on-the-nodes">Deploying dataset on <code>Nodes</code> </a>
 3. <a class="scroller-link" href="#step-3-write-a-federated-model-trainingplan-aggregator-and-strategy" >Write a Federated Model</a>
 4. <a class="scroller-link" href="#step-4-how-to-run-and-monitor-an-experiment">Run and monitor a Federated Model</a>
 5. <a class="scroller-link" href="#step-5-retrieving-the-model-and-performing-model-evaluation">Model retrieval and evaluation</a>


## Step 1: Setting up the `Nodes`.

In order to run Fed-BioMed, you need to start first one or several `Nodes`. When starting the `Nodes`, each of them will try to connect to the `Researcher`, as shown in the diagram below (Diagram 1). The connection will fail if the researcher component is not up and running, and the nodes will retry to connect every 2 seconds until they create a successful connection with the researcher server. 

!!! note "Nodes have only out-bound requests"
    Nodes only send out-bound connection requests and doesn't have in-bound connections. Shortly, nodes are client rather than a server. 


![step-1-basic-fedbiomed-architecture](../assets/img/diagrams/fedbiomed-base-architecture.jpg#img-centered-xlr)
*Diagram 1: `Nodes` in Fed-BioMed, and long polling RPC calls to researcher server.*


## Step 2: Deploying a dataset on the `Nodes`.

The nodes store the datasets locally in the file system where they run. Each dataset needs to be registered/deployed using the Node GUI or Node CLI. This process identifies the dataset to be able to use in training. 

### Step 2.1: Loading a dataset into a `Node`.

Fed-BioMed supports standard data sources, such as .csv files and image folders, and provides specific tools for loading medical data formats, such as medical imaging, signals and genomics information (Diagram 2).

![step-2-loading-data](../assets/img/diagrams/fedbiomed-workflow-step2-loading_data.jpg#img-centered-lr)
*Diagram 2: loading data into a `Node`. Different data types are available, especially for medical datasets.*


After a dataset is deployed, `Node` will be able to train the models submitted by the `Researcher`. The user (researcher) must specify the tag that identifies the dataset. Please refer to Diagram 4 to see the example of dataset identifiers as well as the tags associated. 

![node-researcher-arch](../assets/img/diagrams/fedbiomed-base-arch-data.jpg#img-centered-xlr)
*Diagram 3: <code>Nodes</code> with respective datasets loaded.*


### Step 2.2: Retrieving `Nodes` dataset information on the `Researcher` side.

It is possible for the `Researcher` to obtain information about the dataset of each `Node`, as shown in the diagram 4 below.
![fedbiomed-data-management](../assets/img/diagrams/fedbiomed-arch-dataset-list.jpg#img-centered-xlr)
*Diagram 4: `Node` datasets information that `Researcher` can retrieve. The researcher can access datasets' metadata such as datasets name, dataset data_type, dataset tags, description and shape stored on each node.*

## Step 3: Write a federated Model (`TrainingPlan`, `Aggregator` and `Strategy`)

To create a Federated Model `Experiment` in Fed-BioMed, three principal ingredients must be provided:

1. a `Training Plan`, which is basically a Python class, containing the model definition and related objects, such as cost function and optimizer, and eventually methods for pre-processing (e.g., data standardization and/or imputation), and post-processing (run after that the training of the model on the `Node` is completed).
2. an `Aggregator` that defines how the model parameters obtained on each node after training are aggregated once received by the `Researcher`. Examples of `Aggregator` can be [`FedProx`](https://arxiv.org/abs/1812.06127) or [`SCAFFOLD`](https://arxiv.org/abs/1910.06378).
3. a `Strategy` that handles both node sampling and node management (e.g., how to deal with non responding nodes). 

![fedbiomed-workflow](../assets/img/diagrams/fedbiomed-workflow-step3.jpg#img-centered-lr)
*Diagram 5: the ingredients needed to train a Federated Model in Fed-BioMed.*


## Step 4: How to run and monitor an `Experiment`

### Running an `Experiment`

The following Diagram 6 provides a technical description of the training process within Fed-BioMed:

1. After the nodes are started they sent constant request to the researcher in order to retrieve the training task (request).  

2. The user/researcher creates and experiment with training plan and the required arguments for the training round. 
3. The global model along with the training arguments is packaged as `TrainRequest` and sent to the `Nodes`. The package contains the training plan, experiment info, dataset identifier (tag) and the arguments for the training round. 
4. Each `Node` trains the model on the data locally stored.
5. The resulting optimized local models are sent back to the `Researcher` as `TrainReply`;
6. The shared local models are aggregated to form a new aggregated global model using the `Aggregator`.

![running-an-experiment-with-fedbiomed](../assets/img/diagrams/fedbiomed-workflow.jpg#img-centered-xlr)
*Diagram 6: Showcasing an iteration of federated training in Fed-BioMed.*

![running-an-experiment-with-fedbiomed-alternate](../assets/img/fedbiomed-workflow-convert.gif#img-centered-lr)
*Diagram 7: Alternate view of an iteration of federated training in Fed-BioMed.*

### Monitoring an `Experiment`.

The loss evolution is sent back to the `Researcher` at each evaluation step during the training. The `Researcher` can keep track of the loss using [Tensorboard](../user-guide/researcher/tensorboard.md), as shown in Diagram 8.

![monitoring-experiment](../assets/img/diagrams/fedbiomed-monitoring.jpg#img-centered-xlr)
*Diagram 8: model training monitoring facility available in Fed-BioMed*

## Step 5: retrieving the model and performing model evaluation

Once federated training is complete, the `Researcher` can retrieve the final global model, as well as other relevant information such as the timing between each connection, loss and the testing metrics value (if a validation dataset is provided). Fed-BioMed provides a number of standard metrics, such as accuracy for classification, or mean squared error for regression, and allows the definition of custom ones. 

![experiment-model](../assets/img/diagrams/fedbiomed-workflow-step5-retrieving-results.jpg#img-centered-lr)
*Diagram 9: model and results collected after training a model using Fed-BioMed framework.*

## Going Further

[**Installation Guide**](../tutorials/installation/0-basic-software-installation.md) 

Detailed steps on how to install Fed-BioMed on your computer.

[**Tutorials**](../tutorials/pytorch/01_PyTorch_MNIST_Single_Node_Tutorial.ipynb) 

More tutorials, examples and how-to.

[**`Nodes` configuration Guide**](../user-guide/nodes/configuring-nodes.md) 

Provides an exhaustive overview of Fed-BioMed `Nodes`.

[**`Researcher` configuration Guide**](../user-guide/researcher/training-plan.md) 

Provides additional info on Fed-BioMed `Researcher`.
