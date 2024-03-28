---
title: Deploying Datasets in Nodes
description: Deploying datasets in nodes makes the datasets ready for federated training with Fed-BioMed.
keywords: fedbiomed configuration,node configuration,deployin datasets
---

# Deploying Datasets in Nodes

Deploying datasets in nodes makes the datasets ready for federated training with Fed-BioMed [experiments](../researcher/experiment.md). 
Dataset deployment in Fed-BioMed essentially means providing metadata for a dataset.
One node can deploy multiple datasets.
Once deployed, the dataset's metadata is saved into the node's database for persistent storage, even after the node is
stopped and restarted. 

Each dataset should have the at least following attributes:

- **Database Name:** A user-readable short name of the dataset for display purposes.
- **Description:** A longer description giving more details about the specifics of each dataset.
- **Tags:** A unique identifier used by the federated training process to identify each dataset.

## Requirements

Fed-BioMed does not support downloading datasets from remote sources, 
except for the default `MNIST` and `MedNIST` datasets.
Therefore, before adding a dataset into a node, please make sure that you already prepared your dataset 
and saved it on the file system. 

## Adding a dataset using the Fed-BioMed CLI

Use the following command to add a dataset into the node identified by the `config-n1.ini` file. 

``` shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node --config config-n1.ini dataset add
```
 
Afterward, you will see the following screen in your terminal.

```
# Using configuration file: config_node.ini #
Welcome to the Fed-BioMed CLI data manager
Please select the data type that you're configuring:
	1) csv
	2) default
	3) mednist
	4) images
	5) medical-folder
	6) flamby
select:
```

It asks you to select what kind of dataset you would like to add. 
The `default` and `mednist` option are configured to automatically download and add the MNIST and MedNIST datasets respectively.
To deploy your own data, you should select `csv`, `image`, `medical-folder` or `flamby` option according to your needs. 
After you select an option, you will be prompted with additional questions that cover both common and option-specific details.

For example, let's suppose that you are going to add a CSV dataset. 
To do that you should type 1 and press enter.
The interface will ask you to insert the common elements: name, tags, and description.

```
Name of the database: My Dataset
Tags (separate them by comma and no spaces): #my-csv-data,#csv-dummy-data
Description: Dummy CSV data
```

If a graphical interface is available, the next step opens a file browser window and asks you to select your
csv file.
If a graphical interface is not available, you will be prompted to insert the full path to the file.

After selecting the file, you will be shown the details of your dataset.

```
Great! Take a look at your data:
name        data_type    tags                                 description     shape      path                                   dataset_id                                    dataset_parameters
----------  -----------  -----------------------------------  --------------  ---------  -----------------  --------------------------------------------  --------------------
My Dataset  csv          ['#my-csv-data', '#csv-dummy-data']  Dummy CSV data  [300, 20]  /path/to/your.csv  dataset_<id>
```

You can also check the list of deployed datasets by using the following command:

```shell
$ ${FEDBIOMED_DIR}/scripts/fedbiomed_run node --config config-n1.ini dataset list
```

It will return the datasets saved into the node identified by the `config-n1.ini` file.

## How to Add Another Dataset to the Same Node

Nodes can store multiple datasets. 
You can follow the previous steps as many times as needed to add other datasets. 

!!! info "Adding datasets with the same path"
    Using the same files or the same path for multiple datasets is allowed, provided that the tags are unique.

!!! warning "Conflicting tags between datasets"
    Tags from one dataset cannot be equal to, or a subset of, the tags of another dataset

For example, CLI on a node:

* accepts to register dataset1 with tags `[ 'tag1', 'tag3' ]`, dataset2 with tags `[ 'tag1', 'tag2' ]` and dataset3 with tags `[ 'tag2', 'tag3' ]`
* refuses to register dataset1 with tags `[ 'tag1', 'tag2' ]` if dataset2 with tags `[ 'tag1' ]` already exists
* refuses to register dataset1 with tags `[ 'tag1', 'tag2' ]` if dataset2 with tags `[ 'tag1', 'tag2', 'tag3' ]` already exists