---
title: Deploying Datasets in Nodes
description: Deploying datasets in nodes makes the datasets ready for federated training with Fed-BioMed.
keywords: fedbiomed configuration,node configuration,deployin datasets
---

# Deploying Datasets in Nodes

Deploying datasets on the nodes prepares them for federated training within Fed-BioMed [experiments](../researcher/experiment.md). This process involves providing metadata for each dataset, enabling the system to understand its characteristics. A single node can host multiple datasets. Once deployed, the dataset's metadata is securely stored in the node's database, ensuring persistence even after restarts or shutdowns.

Each dataset must include at least the following attributes:

- **Database Name:**  A concise, user-friendly name for the dataset used for display purposes.
- **Description:** A detailed explanation providing further context and information about the dataset's content and origin.
- **Tags:** Unique identifiers used by the federated training process to distinguish datasets accurately.

Fed-BioMed does not support downloading datasets from remote sources, except for the default `MNIST` and `MedNIST` datasets. Therefore, before adding a dataset into a node, please make sure that you already prepared your dataset and saved it on the file system.

## Adding a dataset using the Fed-BioMed CLI

Use the following command to add a dataset into the node located in the directory `./my-node`

``` shell
$ fedbiomed node --path my-node dataset add
```

After given the permission the create component `my-node`, you will see the following screen in your terminal.

```
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

The interface prompts you to select the type of dataset you would like to add. The `default` and `mednist` options are preconfigured to automatically download and add the MNIST and MedNIST datasets. To deploy your own data, you can select one of the following options: `csv`, `image`, `medical-folder`, or `flamby`, based on your requirements. After selecting an option, you will be prompted to provide additional details, covering both common and option-specific attributes.

For example, suppose you want to add a CSV dataset. To do this, type `1` and press Enter. The interface will then ask you to provide the common attributes: dataset name, tags, and description.

!!! note "Falmby"
    To enable option 6 in Fed-BioMed, you'll need to install FLamby as an external dependency. After installing Fed-BioMed, use the following command: `pip install git+https://github.com/owkin/FLamby@main`."

```
Name of the database: My Dataset
Tags (separate them by comma and no spaces): #my-csv-data,#csv-dummy-data
Description: Dummy CSV data
```

If a graphical interface is available, the next step opens a file browser window and asks you to select your csv file. If a graphical interface is not available, you will be prompted to insert the full path to the file.

After selecting the file, you will be shown the details of your dataset.

```
Great! Take a look at your data:
name        data_type    tags                                 description     shape      path                                   dataset_id                                    dataset_parameters
----------  -----------  -----------------------------------  --------------  ---------  -----------------  --------------------------------------------  --------------------
My Dataset  csv          ['#my-csv-data', '#csv-dummy-data']  Dummy CSV data  [300, 20]  /path/to/your.csv  dataset_<id>
```

You can also check the list of deployed datasets by using the following command:

```shell
$ fedbiomed node --path my-node dataset list
```

It will return the datasets saved into the node created in the directory `./my-node`.

## How to Add Another Dataset to the Same Node

Nodes can store multiple datasets. You can follow the previous steps as many times as needed to add other datasets.

!!! info "Adding datasets with the same path"
    Using the same files or the same path for multiple datasets is allowed, provided that the tags are unique.

!!! warning "Conflicting tags between datasets"
    Tags from one dataset cannot be equal to, or a subset of, the tags of another dataset

For example, CLI on a node:

* accepts to register dataset1 with tags `[ 'tag1', 'tag3' ]`, dataset2 with tags `[ 'tag1', 'tag2' ]` and dataset3 with tags `[ 'tag2', 'tag3' ]`
* refuses to register dataset1 with tags `[ 'tag1', 'tag2' ]` if dataset2 with tags `[ 'tag1' ]` already exists
* refuses to register dataset1 with tags `[ 'tag1', 'tag2' ]` if dataset2 with tags `[ 'tag1', 'tag2', 'tag3' ]` already exists
