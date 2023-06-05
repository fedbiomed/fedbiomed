---
title: Setting up your Fed-BioMed environment
description: Installation tutorial for Fed-BioMed
keywords: Fed-BioMed, environment, setup
---

# Set up your Fed-BioMed environment

## A word on Fed-BioMed components

A Fed-BioMed instance includes 3 types of components :

* the `network` which handles the communication between the nodes and the researcher. It is composed of a MQTT messaging server and a HTTP/REST file exchange server for the models and parameters
* one or more `nodes`, each one provides datasets for experiments and locally trains the models on these datasets
* the `researcher` which defines and orchestrates a federated learning experiment. The experiment looks for nodes providing expected datasets, selects nodes, sends nodes a model and initial parameters, requests nodes to locally train the model on datasets, collects local training output, federates the output to update aggregated parameters.

`nodes` and `researcher` compose the Fed-BioMed software, `network` are supporting services based on docker images.

In this tutorial you learn how to launch Fed-BioMed components using the `fedbiomed_run` script.


## Launching Fed-BioMed components

### Network

Network is the first Fed-BioMed component to launch, as it enables other components to communicate.

Launch the network with :
```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run network
```

Check the mosquitto and fedbiomed-network containers are `Up` with :

```shell
$ docker container ps
CONTAINER ID  IMAGE                 COMMAND                CREATED       STATUS       PORTS                                                                                NAMES
954fda27350a  fedbiomed/dev-restful "/entrypoint.sh"       4 seconds ago Up 3 seconds 0.0.0.0:8844->8000/tcp, :::8844->8000/tcp                                            fedbiomed-dev-restful
ff56256260b1  eclipse-mosquitto     "/usr/sbin/mosquitto…" 4 seconds ago Up 3 seconds 0.0.0.0:1883->1883/tcp, :::1883->1883/tcp, 0.0.0.0:9001->9001/tcp, :::9001->9001/tcp fedbiomed-dev-mqtt

```


### Node

Once the network is ready, one or more nodes shall be started and configured to provide datasets for the experiments.

By default, a Fed-BioMed node does not provide any dataset to experiments. By adding one or more datasets to a node, you indicate which data a Fed-BioMed node provides to experiments for training future model.

When you stop and restart a node, data sharing configuration is retained : previously added datasets remain available for next experiments.


#### Starting a first node

Launch a node with :
````
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start
````

You should get the following output;

```
** Activating fedbiomed-node environment
Conda   env: fedbiomed-node
Python  env: /path/to/python/env/
MQTT   host: localhost
MQTT   port: 1883
UPLOADS url: http://localhost:8844/upload/
2022-01-06 11:40:43,303 fedbiomed INFO - Component environment:
2022-01-06 11:40:43,303 fedbiomed INFO - - type                             = ComponentType.NODE
2022-01-06 11:40:43,303 fedbiomed INFO - - training_plan_approval           = False
2022-01-06 11:40:43,303 fedbiomed INFO - - allow_default_training_plans     = True


   __         _ _     _                          _                   _
  / _|       | | |   (_)                        | |                 | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |  _ __   ___   __| | ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` | | '_ \ / _ \ / _` |/ _ \
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| | | | | | (_) | (_| |  __/
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_| |_| |_|\___/ \__,_|\___|



	- 🆔 Your node ID: <generated_node_id>

2022-01-06 11:40:43,852 fedbiomed INFO - Node started as process with pid = 822849
To stop press Ctrl + C.
2022-01-06 11:40:43,853 fedbiomed INFO - Launching node...
2022-01-06 11:40:43,853 fedbiomed WARNING - Training plan approval for train request is not activated. This might cause security problems. Please, consider to enable training plan approval.
2022-01-06 11:40:43,853 fedbiomed INFO - Starting communication channel with network
2022-01-06 11:40:43,855 fedbiomed INFO - Messaging <generated_node_id> connected to the message broker, object = <fedbiomed.common.messaging.Messaging object at 0x7ff6b39f3070>
2022-01-06 11:40:43,866 fedbiomed DEBUG -  adding handler: MQTT
2022-01-06 11:40:43,866 fedbiomed INFO - Starting task manager
```

  

#### Adding data to node

Starting a new node doesn't provide any data to Fed-BioMed experiments through the node.
Adding a dataset is the process of indicating to a Fed-BioMed node which dataset you wish to make available for Fed-BioMed experiments through the node.
Same datasets will be automatically provided at subsequent starts of the node.


Begin adding a dataset to the node with this command :
```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node add

```

The command then asks you the type of dataset to add to the node :

```
Welcome to the Fed-BioMed CLI data manager
Please select the data type that you're configuring:
       1) csv
       2) default
       3) images
select:

```

Choose `2) default` to download the MNIST dataset and add it to the node. Other options allow you to add custom datasets formatted as CSV files or image banks.

The command then asks for the tags that the node uses to advertise datasets to experiments :

```
MNIST will be added with tags ['#MNIST', '#dataset'] [y/N]
```

Choose `y` to confirm proposed default tags.

A file browser opens : select the folder where you want to save the downloaded MNIST dataset. If you re-use the same folder for next tests, the cached version of MNIST will be used.

When downloading completes you should have the following output :
```
Great! Take a look at your data:
name    data_type   tags                     description        shape                path                dataset_id
------  ---------   ----------------------   ----------------   ------------------   -------------       ---------------------
MNIST   default     ['#MNIST', '#dataset']   MNIST database     [60000, 1, 28, 28]   <data_set_path>     <dataset_id>

```

You can check at any time which datasets are provided by a node with :
```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node list
```

When no dataset is provided by the node, the command `${FEDBIOMED_DIR}/scripts/fedbiomed_run node list` will answer `No data has been set up.` as its final output line.


#### Starting more nodes

To launch and configure more than one node, specify a different (non-default) configuration file for all commands related a the subsequent node.

For example to launch and add a dataset to a second node using the `config2.ini` configuration file :
```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini start
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini list
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini add
```

**Warning : if you launch more than one node with the same configuration file, no error is detected, but the nodes are not functional**


### Researcher

Once the network and nodes are ready, you can start working with the researcher.

Launch the researcher jupyter notebook console with :
```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher start
```

For next tutorials, create a new notebook (`New` and `Python3` in the top right corner of the jupyter notebook), and cut/paste the tutorial code snippets in the notebook.

Several example notebooks are also provided with Fed-BioMed.


## Clean and restart Fed-BioMed components

### A word on working with environments

This tutorial explained how to launch Fed-BioMed components using the `fedbiomed_run` script.
Behind the hood, each Fed-BioMed component runs in its own environment (conda, variables).

If at some point you want to work interactively in the same environment as a Fed-BioMed component
(eg. for debugging), you can activate this environment from a console.

**Warning :** this feature only works with **bash**, **ksh** and **zsh** shells (other shells like csh/tcsh are not yet suppported)

To activate the **network** environment:

```
$ source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment network
```

To activate the **node** environment:

```
$ source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment node
```

To activate the **researcher** environment:

```
$ source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment researcher
```

You can also, reset the environment with:

```
$ source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment reset
```

## Clean

A Fed-BioMed instance can handle successive operations like adding and then removing nodes or datasets, conducting sequential experiments.
But after testing and tweeking, thing may get wrong. At this point, we provide you a script to clean all things.
Afterwards, you will need to restart from scratch (start network, add datasets to nodes, start nodes, etc...)

To clean your Fed-BioMed instance :

* stop the researcher : shutdown the notebook kernel (`Quit` in on the notebook interface or `ctrl-C` on the console)
* stop the nodes : interrupt (`ctrl-C`) on the nodes console
* stop the network and remove all configuration files, dataset sharing configuration, temporary files, caches for all Fed-BioMed components with :

```
$ source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
```

When you restart a node after cleaning the Fed-BioMed instance, the node doesn't provides any dataset, as the dataset sharing configuration was reset in the cleaning process. Of course, Fed-BioMed did not delete any data, it just stopped sharing them.


## Restart

After cleaning your Fed-BioMed environment, restart the network, a node and the researcher to be ready for the next tutorial ... do you remember the commands ?

```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run network
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node add
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node start
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher start
```

## What's Next?

You now have a network and a node ready for an experiment. You also know how to stop an experiment, clean and restart your Fed-BioMed environment. In the following tutorial you will launch your first Fed-BioMed experiment.
