---
title: Federated Training architecture
description: This page presents the architecture of Fed-BioMed.
keywords: Federated Learning, architecture
---

# Fedbiomed Main components

Fed-BioMed has three main components ensuring the correct execution of federated learning algorithms.These components 
are `Node`, `Researcher` and `Network`, and are defined in the following sections:

### `Node`

In Fed-BioMed, `Node` provides 2 main functionalities: 

 - `Node` stores datasets, upon which the Federated Learning Model will be trained. Datasets paths towards files and folders in a [TinyDB](https://tinydb.readthedocs.io/en/latest/) database, as well as other metadata such as datatype, data shape, ... .
 - `Node` trains a model upon a `TrainRequest` (sent by `Researcher`), and send it back to the `Researcher` once training is completed.
 
`Node` is a client that is responsible for sending replies in response to Researcher requests. Since `Node` is not a 
server, communication between node and researcher is provided through `Network` component. When a node is started,  
it directly connects to the `Network` component.

More details about `Node` installation can be found [here](../user-guide/nodes/configuring-nodes.md).

### `Researcher`

In Fed-BioMed, a `Researcher` is an entity that defines the Federated Learning model. In order to define a model, `Researcher` has to provide 3 elements:

- a `TrainingPlan`(containing a model, method to load/pre-process data and dependencies)
- a `Strategy`, which defines how nodes are selected / sampled while training a Federated Mode
- an `Aggregator`, which purpose is to aggregate the local model coming from each node into an aggregated model. Aggregation is performed on `Researcher` side. `Researcher` can be run using plain Python scripts or Jupyter Notebook (thus in an interactive fashion).

`Researcher` orchestrates the training, by sending Requests to `Nodes`. Thus, for each Request sent by the `Researcher`, 
`Node` must answer back with an appropriate Reply.

More details about `Researcher` and its installation can be found [here](../user-guide/researcher/aggregation.md).

### `Network`

`Network` is an entity connecting researcher to the nodes. It provides a Restful HTTP server used for sending TrainingPlan 
and model weights between `Nodes` and `Researcher`, and a MQTT server for short and fast message and `Requests` exchange 
between `Node` and `Researcher`. Furthermore, `MQTT` server enables each `Node` to communicate with other `Nodes` 
without passing by the `Researcher`, permitting much more flexibility than other communication protocols such as gRPC. 

`Network` should work as a Trusted Third Party when considering advanced security options & protocols. More details about its configuration and deployment can be found [here](../tutorials/installation/1-setting-up-environment.md).

## Fed-BioMed Architecture

Relationship between each main component aforementioned is detailed in the figure below:

![alt text](../assets/img/diagrams/fedbiomed-base-arch-data-train.jpg#img-centered)
*Fed-BioMed Architecture, with three main components: Nodes; containing datasets to be   used for training models, 
Researcher; running the training of the model and Network; connecting Nodes to Researcher.*

As shown in the diagram, `Network` is a central component in Fed-BioMed, that links `Nodes` to `Researcher`, and ensures 
message and files delivery. `Nodes` are in charge of running the model sent by the `Researcher`, and send the resulting 
trained model to the `Researcher`. Large files such as `TrainingPlan` and model parameters are exchanged over a Restful 
HTTP server whereas messages, Requests and Replies are sent through a MQTT server.

## `Network` configuration

For information on how to configure `Network`, 
please [follow `Network` configuration steps](../tutorials/installation/1-setting-up-environment.md)

## `Node` configuration

For information on how to configure `Node`, 
please [follow `Node` configuration steps](../user-guide/nodes/configuring-nodes.md)

## `Researcher` configuration

For information on how to configure `Researcher`, 
please [follow `Researcher` configuration steps](../user-guide/researcher/experiment.md)

