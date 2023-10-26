---
title: Deploying Datasets in Nodes
description: Deploying datasets in nodes makes the datasets ready for federated training with Fed-BioMed.
keywords: fedbiomed configuration,node configuration,deployin datasets
---

# Deploying Datasets in Nodes

Deploying datasets in nodes makes the datasets ready for federated training with Fed-BioMed [experiments](../researcher/experiment.md). It is the process of indicating metadata of the dataset. Thanks to that, node can access data and perform training based on given arguments in the experiment. A node can deploy multiple datasets and train models on them. Currently, Fed-BioMed supports adding CSV and Image datasets into nodes. It adds data files from the file system and saves their locations into the database. Each dataset should have the following attributes;

- **Database Name:** The name of the dataset
- **Description:** The description of the dataset. Thanks to that researchers can understand what the dataset is about.
- **Tags:** This attribute identifies the dataset. It is important because when the experiment is initialized for training it searches the nodes based on given tags.


## Adding Dataset with Fed-BioMed CLI

You can use Fed-BioMed CLI to add a dataset into a node. You can either configure a new node by adding new data or you can use a node which has been already configured. The config files in the `etc` directory of Fed-BioMed corresponds to the nodes that have already been configured.

The following code will add the dataset into the node configured using the `config-n1.ini` file. However, if there is no such config file, this command will automatically create a new one with a given file name which is `config-n1.ini`.

``` shell

$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config-n1.ini add

```

Another option is to add a dataset into the default node without addressing any config file. In this case, a default config file for the node is created.

```
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node add
```

**Note:** Adding dataset into nodes with config is better when you work with multiple nodes.


Before adding a dataset into a node, please make sure that you already prepared your dataset and saved it on the file system. Adding a dataset doesn't mean that you can add it from remote sources. The data should be in the host machine.

Please open a terminal and run one of the commands above. Afterward, you will see the following screen in your terminal.

```shell
Conda   env: fedbiomed-node
Python  env: /path/to/your/fedbiomed/
UPLOADS url: http://localhost:8844/upload/


   __         _ _     _                          _                   _
  / _|       | | |   (_)                        | |                 | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |  _ __   ___   __| | ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` | | '_ \ / _ \ / _` |/ _ \
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| | | | | | (_) | (_| |  __/
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_| |_| |_|\___/ \__,_|\___|

        - 🆔 Your node ID: node_<id_of_the_node>

Welcome to the Fed-BioMed CLI data manager
Please select the data type that you're configuring:
        1) CSV
        2) default
        3) mednist
        4) images
select:
```

It asks you to select what kind of dataset you would like to add. As you can see, it offers four options, namely `csv`, `default`, `mednist`, and `images`. 
The `default` and `mednist` option are configured to automatically downloading and adding the MNIST and MedNIST datasets respectively.
To configure your data, you should select `csv` or `image` option according to your needs. Let's suppose that you are going to add a CSV dataset. To do that you should type 1 and press enter.

```shell
Name of the database: My Dataset
Tags (separate them by comma and no spaces): #my-csv-data,#csv-dummy-data
Description: Dummy CSV data
```

As you can see, it asks for the information which are necessary to save your data. For the `Tags` part, you can enter only one tag or more than one. Please make sure that you use a comma to separate multiple tags (without space). Afterward, it will open a browser window and ask you to select your dataset. After selecting, you will see the details of your dataset.

```shell
Great! Take a look at your data:
name        data_type    tags                                 description     shape        path               dataset_id
----------  -----------  -----------------------------------  ----------      ---------    ---------------    -----------
My Dataset  csv          ['#my-csv-data', '#csv-dummy-data']  Dummy CSV data  [1000, 15]   /pat/to/your.csv   dataset_<id>
```

You can also check the list of datasets by using the following command:

```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config-n1.ini list
```

It will return the datasets saved into the node which use the `config-n1.ini` file as config.

## How to Add Another Dataset to the Same Node

As mentioned, nodes can store multiple datasets. You can follow the previous steps to add another dataset. 
While adding another dataset, the config file of the node should be indicated. Otherwise, CLI can create or use 
default node config to deploy your dataset. It is allowed to add datasets 
with the same file path, or name. As the tags are used as an identifier for the dataset, the CLI checks 
they are not conflicting with another dataset on the same node.

!!! info "Conflicting tags between datasets"
    Tags from two datasets on the same node need to respect some rules:

    * tags from one dataset cannot be a **subset** of the tags of another dataset
    * tags from one dataset cannot be a **superset** of the tags of another dataset

    As a consequence, two datasets on the same node cannot have exactly the same tags.

For example, CLI on a node:

* accepts to register dataset1 with tags `[ 'tag1', 'tag3' ]`, dataset2 with tags `[ 'tag1', 'tag2' ]` and dataset3 with tags `[ 'tag2', 'tag3' ]`
* refuses to register dataset1 with tags `[ 'tag1', 'tag2' ]` and dataset2 with tags `[ 'tag1' ]`
* refuses to register dataset1 with tags `[ 'tag1', 'tag2' ]` and dataset2 with tags `[ 'tag1', 'tag2', 'tag3' ]`