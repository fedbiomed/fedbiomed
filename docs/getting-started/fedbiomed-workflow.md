---
title: Federated Training Workflow
description: This page presents the workflow of Fed-BioMed.
keywords: Federated Learning, workflow
---

# Fed-BioMed Workflow

We present in the following page a short step-by-step illustration detailing the workflow of Fed-BioMed.

The steps are:

 1. <a class="scroller-link" href="#step-1-setting-up-network-and-nodes">Setting up the <code>Network</code> and <code>Nodes</code></a>
 2. <a class="scroller-link" href="#step-2-deploying-datasets-on-the-nodes">Deploying dataset on <code>Nodes</code> </a>
 3. <a class="scroller-link" href="#step-3-write-a-federated-model-trainingplan-aggregator-and-strategy" >Write a Federated Model</a>
 4. <a class="scroller-link" href="#step-4-how-to-run-and-monitor-an-experiment">Run and monitor a Federated Model</a>
 5. <a class="scroller-link" href="#step-5-retrieving-the-model-and-performing-model-evaluation">Model retrieval and evaluation</a>

For installation, please [visit the software installation page](../tutorials/installation/0-basic-software-installation.md).

## Step 1: Setting up `Network` and `Nodes`.

In order to run Fed-BioMed, you need to start first the `Network` component and then one or several `Nodes`.
When setting up the `Nodes`, each of them will connect to the `Network`, as shown in the diagram below (Diagram 1). 

![step-1-basic-fedbiomed-network](../assets/img/diagrams/fedbiomed-worflow-step1.jpg#img-centered-lr)
*Diagram 1: `Nodes` and `Network` in Fed-BioMed, and their interactions with the other components.*


## Step 2: Deploying a dataset on the `Nodes`.

### Step 2.1: Loading a dataset into a `Node`.

Fed-BioMed supports standard data sources, such .csv files and image folders, and provides specific tools for loading medical data formats, such as  medical imaging, signals and genomics information (Diagram 2).

![step-2-loading-data](../assets/img/diagrams/fedbiomed-workflow-step2-loading_data.jpg#img-centered-lr)
*Diagram 2: loading data into a `Node`. Different data types are available, especially for medical datasets. Folder and Genes icons courtesy of Freepik, Clinical icon courtesy of Parzival' 1997, Flaticon.


Once provided with a dataset, a `Node` is able to train the models sent by the `Researcher`.

![node-researcher-arch](../assets/img/diagrams/fedbiomed-worklow-step-2-dataset.jpg#img-centered-lr)
*Diagram 3: <code>Nodes</code> with respective datasets loaded.*


### Step 2.2: Retrieving `Nodes` dataset information on the `Researcher` side.

It is possible for the `Researcher` to obtain information about the dataset of each `Node`, as shown in the diagram 4 below.
![fedbiomed-data-management](../assets/img/diagrams/fedbiomed-workflow-step2-list-request.jpg#img-centered-lr)
* Diagram 4: `Node` datasets information that `Researcher` can retrieve. The researcher can access datasets' metadata such as datasets name, dataset data_type, dataset tags, description and shape stored on each node.*

## Step 3: Write a federated Model (`TrainingPlan`, `Aggregator` and `Strategy`)

To create a Federated Model `Experiment` in Fed-BioMed, three principal ingredients must be provided:

1. a `Training Plan`, which is basically a Python class, containing the model definition and related objects, such as cost function and optimizer, and eventually methods for pre-processing (e.g., data standardization and/or imputation), and post-processing (run after that the training of the model on the `Node` is completed).
2. an `Aggregator` that defines how the model parameters obtained on each node after training are aggregated once received by the `Researcher`. Examples of `Aggregator` can be [`FedProx`](https://arxiv.org/abs/1812.06127) or [`SCAFFOLD`](https://arxiv.org/abs/1910.06378).
3. a `Strategy` that handles both node sampling and node management (e.g., how to deal with non responding nodes). 

![fedbiomed-workflow](../assets/img/diagrams/fedbiomed-workflow-step3.jpg#img-centered-lr)
*Diagram 5: the ingredients needed to train a Federated Model in Fed-BioMed.*


## Step 4:  how to run and monitor an `Experiment`

### Running an `Experiment`

The animation of Diagram 6 shows how a federated model is trained within Fed-BioMed: 

1. The global model is sent to the `Nodes` through the `Network`. The model's architecture is defined in a `TrainingPlan`, and weights are contained in a specific file exchanged over the `Network`;
2. Each `Node` trains the model on the available local data;
3. The resulting optimized local models are sent back to the `Researcher` through the `Network`;
4. The shared local models are aggregated to form a new aggregated global model, according to the `Aggregator`.

![running-an-experiment-with-fedbiomed](../assets/img/diagrams/fedbiomed-workflow-step4.gif#img-centered-lr)
*Diagram 6: animation showcasing an iteration of federated training in Fed-BioMed. The model defined in a `TrainingPlan` is sent to the `Nodes` and trained on their local data, to be subsequently aggregated. Grayed-out models represent an untrained model, while colored ones represent a model trained on local data.*

Diagram 7 provides a more technical description of the training process within Fed-BioMed:


1. The `Researcher` initiates training round by issuing to the `Nodes` a TrainRequest through MQTT component of the `Network`, and by sending the model and the current global parameters through the Restful server component of the `Network`. 
2. Upon a TrainRequest, each `Node` trains the model and issues a TrainingReply to the Researcher passing through the MQTT of the `Network`, as well as the updated parameters (through the Restful server)
3. Once updated, the model parameters coming from each Node are collected by the `Researcher`, and aggregated to create the new global model.

![fedbiomed-training-process](../assets/img/diagrams/fedbiomed-workflow-summary.jpg#img-centered-lr)
*Diagram 7: details of the Requests and Replies sent to each components when performing one round of training.* 

### Monitoring an `Experiment`.

The loss evolution is sent back to the `Researcher` at each evaluation step during training. The `Researcher` can keep track of the loss using [Tensorboard](../user-guide/researcher/tensorboard.md), as shown in DIagram 8.

![monitoring-experiment](../assets/img/diagrams/fedbiomed-workflow-step4-monitoring.jpg#img-centered-lr)
*Diagram 8: model training monitoring facility available in Fed-BioMed*

## Step 5: retrieving the model and performing model evaluation

Once federated training is complete, the `Researcher` can retrieve the final global model, as well as other relevant information such as the timing between each connection, loss and the testing metrics value (if a validation dataset is provided). Fed-BioMed provides a number of standard metrics, such as accuracy for classification, or mean squarred error for regression, and allows the definition of custom ones. 

![experiment-model](../assets/img/diagrams/fedbiomed-workflow-step5-retrieving-results.jpg#img-centered-lr)
*Diagram 9: model and results collected after training a model using Fed-BioMed framework.  
Icons courtesy of Ramy W. - Flaticon*

## Going Further

[**Installation Guide**](../tutorials/installation/0-basic-software-installation.md) 

Detailed steps on how to install Fed-BioMed on your computer.

[**Tutorials**](../tutorials/pytorch/01_PyTorch_MNIST_Single_Node_Tutorial.ipynb) 

More tutorials, examples and how-to.

[**`Nodes` configuration Guide**](../user-guide/nodes/configuring-nodes.md) 

Provides an exhaustive overview of Fed-BioMed `Nodes`.

[**`Researcher` configuration Guide**](../user-guide/researcher/training-plan.md) 

Provides additional info on Fed-BioMed `Researcher`.
