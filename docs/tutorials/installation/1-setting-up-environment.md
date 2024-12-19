---
title: Setting up your Fed-BioMed environment
description: Installation tutorial for Fed-BioMed
keywords: Fed-BioMed, environment, setup
---

# Set up your Fed-BioMed environment

## A word on Fed-BioMed components

A Fed-BioMed instance includes 2 types of components :

* one or more `nodes`, each one provides datasets for experiments and locally trains the models on these datasets
* the `researcher` which defines and orchestrates a federated learning experiment. The experiment looks for nodes providing expected datasets, selects nodes, sends nodes a model and initial parameters, requests nodes to locally train the model on datasets, collects local training output, federates the output to update aggregated parameters.

In this tutorial you learn how to launch Fed-BioMed components using the `fedbiomed_run` script.


## Launching Fed-BioMed components

### Node

one or more nodes need to be started and configured to provide datasets for the experiments. By default, a Fed-BioMed node does not share any dataset for experiments. Adding one or more datasets to a node specifies the data that the node will make available for experiments to train future models.

When you stop and restart a node, the data sharing configuration is retained: previously added datasets remain available for subsequent experiments.

#### Starting a first node

Starting a node using the command below will create a default component directory if it does not already exist. All component-related configurations and files will be stored in this directory, which will be created in the location where the command is executed. The default component folder name for node is `fbm-node`.

````
$ fedbiomed node start
````


You can specify a different directory using the `--path` or `-d` option. If a component already exists in the specified directory, the CLI will use that component; otherwise, it will create a new one in the given directory. The directory path can be either relative or absolute.

```
$ fedbiomed node --path <path-to-my-component> start
```

After executing any command to start a node, you should see the following output:

```
2023-10-20 10:20:09,889 fedbiomed DEBUG - Reading default biprime file "biprime0.json"


   __         _ _     _                          _                   _
  / _|       | | |   (_)                        | |                 | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |  _ __   ___   __| | ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` | | '_ \ / _ \ / _` |/ _ \
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| | | | | | (_) | (_| |  __/
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_| |_| |_|\___/ \__,_|\___|



	- ID Your node ID: <node_id>

2023-10-20 10:20:12,658 fedbiomed INFO - Node started as process with pid = <pid>
To stop press Ctrl + C.
2023-10-20 10:20:12,659 fedbiomed INFO - Launching node...
2023-10-20 10:20:12,659 fedbiomed WARNING - Training plan approval for train request is not activated. This might cause security problems. Please, consider to enable training plan approval.
2023-10-20 10:20:12,659 fedbiomed INFO - Starting communication channel with network
2023-10-20 10:20:12,659 fedbiomed DEBUG -  adding handler for: GRPC
2023-10-20 10:20:12,660 fedbiomed INFO - Starting task manager
2023-10-20 10:20:12,660 - Not able to send log message to remote party
2023-10-20 10:20:12,662 fedbiomed INFO - Starting task listeners
2023-10-20 10:20:12,662 fedbiomed DEBUG - Sending new task request to researcher
2023-10-20 10:20:12,664 fedbiomed INFO - Researcher server is not available, will retry connect in 2 seconds
2023-10-20 10:20:14,666 fedbiomed DEBUG - Sending new task request to researcher
2023-10-20 10:20:14,667 fedbiomed INFO - Researcher server is not available, will retry connect in 2 seconds
```

#### Adding data to node

Starting a new node doesn't provide any data to Fed-BioMed experiments through the node.
Adding a dataset is the process of indicating to a Fed-BioMed node which dataset you wish to make available for Fed-BioMed experiments through the node.
Same datasets will be automatically provided at subsequent starts of the node.

Begin adding a dataset to the node with this command :
```
$ fedbiomed node dataset add

# or explicitly specify the component directory
# fedbiomed node --path fbm-node dataset add

```

The command then asks you the type of dataset to add to the node :

```
Welcome to the Fed-BioMed CLI data manager
Please select the data type that you're configuring:
       1) csv
       2) default
       3) images
       ...
select:

```

Choose `2) default` to download the MNIST dataset and add it to the node. Other options allow you to add custom datasets formatted as CSV files, image banks, BIDS-like medical-folder, etc.

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
$ fedbiomed node dataset list
```

When no dataset is provided by the node, the command `fedbiomed node dataset list` will answer `No data has been set up.` as its final output line.


#### Starting more nodes

To launch and configure more than one node, specify a different (non-default) configuration file for all commands related to the subsequent node. For example, to launch and add a dataset to a second node, replace <path-to-my-second-component> with a relative or absolute path to the component folder. The folder will be created if it does not already exist.

```
$ fedbiomed node --path <path-to-my-second-component> start
$ fedbiomed node -d <path-to-my-second-component> dataset list
$ fedbiomed node -d <path-to-my-second-component>  dataset add
```

**Warning : if you launch more than one node with the same directory specification, no error is detected, but the nodes are not functional**


### Researcher

Once the nodes are ready, you can start working with the researcher.

Launch the researcher jupyter notebook console with :
```
$ fedbiomed researcher start
```

For next tutorials, create a new notebook (`New > Notebook` and `Python3 kernel` in the top right corner of the jupyter notebook), and cut/paste the tutorial code snippets in the notebook.

Several example notebooks are also provided with Fed-BioMed.


## Clean and restart Fed-BioMed components

### A word on working with environments

This tutorial explained how to launch Fed-BioMed components using the `fedbiomed` command.
Behind the hood, each Fed-BioMed component runs in its own environment (conda, variables).

If at some point you want to work interactively in the same environment as a Fed-BioMed component
(eg. for debugging), you can activate this environment from a console.

**Warning :** this feature only works with **bash**, **ksh** and **zsh** shells (other shells like csh/tcsh are not yet suppported)

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
Afterwards, you will need to restart from scratch (add datasets to nodes, start nodes, etc...)

To clean your Fed-BioMed instance :

* stop the researcher : shutdown the notebook kernel (`Quit` in on the notebook interface or `ctrl-C` on the console)
* stop the nodes : interrupt (`ctrl-C`) on the nodes console
* remove all configuration files, dataset sharing configuration, temporary files, caches for all Fed-BioMed components with :

```
$ source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
```

When you restart a node after cleaning the Fed-BioMed instance, the node doesn't provides any dataset, as the dataset sharing configuration was reset in the cleaning process. Of course, Fed-BioMed did not delete any data, it just stopped sharing them.


## Restart

After cleaning your Fed-BioMed environment, restart a node and the researcher to be ready for the next tutorial ... do you remember the commands ?

```
$ fedbiomed node dataset add
$ fedbiomed node start
$ fedbiomed researcher start
```

## What's Next?

You now have a node and a researcher ready for an experiment. You also know how to stop an experiment, clean and restart your Fed-BioMed environment. In the following tutorial you will launch your first Fed-BioMed experiment.
