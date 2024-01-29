---
title: Federated Training architecture
description: This page presents the architecture of Fed-BioMed.
keywords: Collaborative Learning, Federated Learning, Federated Analytics, architecture
---

# Fedbiomed Main components

Fed-BioMed has two components ensuring the correct execution of collaborative learning algorithms. These components  are `Node` and `Researcher`, and are defined in the following sections:

### `Node`

In Fed-BioMed, `Node` provides 2 main functionalities: 

 - `Node` stores datasets, upon which the Federated Learning Model will be trained and Federated Analytics will be performed. Datasets paths towards files and folders in a [TinyDB](https://tinydb.readthedocs.io/en/latest/) database, as well as other metadata such as datatype, data shape, ... .
 - `Node` trains a model upon a `TrainRequest` (sent by `Researcher`), and send it back to the `Researcher` once training is completed.
 
`Node` is a client that is responsible for sending replies in response to Researcher requests. Since `Node` is not a 
server, communication between node and researcher is provided through `Network` component. When a node is started,  
it directly connects to the `Network` component.

More details about `Node` installation can be found [here](../user-guide/nodes/configuring-nodes.md).

### `Researcher`

In Fed-BioMed, a `Researcher` is an entity that orchestrates federated workflow among the nodes. It is the server (gRPC) that each node participating federated experiment is connected.  `Researcher` component is also the entity that the end-user connect and uses for federated experiment. It provides Python APIs to define the experiment and all the elements needed for an experiment. In order to define an experiment, user has to provide 3 elements using `Researcher` component:

- a `TrainingPlan`(containing a model, method to load/pre-process data and dependencies)
- a `Strategy`, which defines how nodes are selected / sampled while training a Federated Mode
- an `Aggregator`, which purpose is to aggregate the local model coming from each node into an aggregated model. Aggregation is performed on `Researcher` side. `Researcher` can be run using plain Python scripts or Jupyter Notebook (thus in an interactive fashion).

`Researcher` orchestrates the training by submitting training tasks to `Nodes`. Connected nodes subscribe the submitted tasks/request through RPC calls. After the task execution is completed `Node` answers back with an appropriate reply for each task submitted and subscribed by them.

More details about `Researcher` and its installation can be found [here](../user-guide/researcher/aggregation.md).


## Fed-BioMed Architecture

Relationship between each main component aforementioned is detailed in the figure below:

![alt text](../assets/img/diagrams/fedbiomed-base-arch-data-train.jpg#img-centered-xlr)
*Fed-BioMed basic architecture*

As shown in the diagram, `Researcher` is a central component in Fed-BioMed that submits federated queries (model training, federated analytics etc.) to the nodes and, collects the results. The network infrastructure is based on nodes collecting tasks from researcher rather than researcher sending requests to the nodes. The tasks created by end-user is stored by `Researcher` and submitted to nodes once they send task-collect request to the `Researcher` component. `Nodes` are in charge of running the model training or executing any other tasks that is submitted by the `Researcher`, and sending the results back to the `Researcher` by creating a reply request. Large files such as model parameters are exchanged using streaming to avoid memory issues. 


## `Node` configuration

For information on how to configure `Node`, please [follow `Node` configuration steps](../user-guide/nodes/configuring-nodes.md)

## `Researcher` configuration

For information on how to configure `Researcher`, please [follow `Researcher` configuration steps](../user-guide/researcher/experiment.md)

