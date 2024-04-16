# Example Notebooks

This directory contains examples about different main and different functionalities available in Fed-BioMed. It also contains a directory that points to tutorials published in the documentation site. The notebooks in the main directory does not contain details information, however, you can find more detailed examples in the `tutorials` directory.

## Datasets Used in the examples

Examples in the main directory uses different type of dataset. While some dataset are common among the examples, some of them are generated within the example notebooks. The instructions for loading generated datasets are available in the notebooks.   

### Loading MNIST Dataset

1. `{FEDBIOMED_DIR}/scripts/fedbiomed_run node dataset add`
  * Select option 2 (default) to add MNIST to the node.
  * Confirm default tags by hitting "y" and ENTER.
  * Pick the folder where MNIST is downloaded.
  * Data must have been added (if you get a warning saying that data must be unique is because it's been already added)

2. Check that your data has been added by executing `{FEDBIOMED_DIR}/scripts/fedbiomed_run node dataset list`
3. Run the node using `{FEDBIOMED_DIR}/scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. it means you are online.
4. Following the same procedure, you can create a second node for client 2.


### Loading Pseudo Adni Dataset

It is necessary to previously configure a node (at least):
1. `{FEDBIOMED_DIR}/scripts/fedbiomed_run node dataset add`
  * Select option 1 to add a csv file to the node
  * Choose the name, tags and description of the dataset
    * use `#test_data` for the tags
  * Pick the .csv file from your PC, located under `fedbiomed/notebooks/data/CSV` folder (here: [pseudo_adni_mod.csv](./data/CSV/pseudo_adni_mod.csv))
  * Data must have been added
2. Check that your data has been added by executing `{FEDBIOMED}/scripts/fedbiomed_run node dataset list`
3. Run the node using `{FEDBIOMED}/scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. It means you are online.

Note: Notebooks/examples may use different tags. Please use the tags that are relevant to the example you are running.

### Loading Adni Dataset

Note: This dataset is different than "Pseudo Adni Dataset". It contains three different CSV for three nodes.

This dataset a realistic dataset of (synthetic) medical information mimicking the ADNI dataset (http://adni.loni.usc.edu/). The data is entirely synthetic and randomly sampled to mimick the variability of the real ADNI dataset**. The training partitions are available at the following link:

https://drive.google.com/file/d/1R39Ir60oQi8ZnmHoPz5CoGCrVIglcO9l/view?usp=sharing

The federated task we aim at solve is to predict a clinical variable (the mini-mental state examination, MMSE) from a combination of demographic and imaging features. The regressors variables are the following features:

['SEX', 'AGE', 'PTEDUCAT', 'WholeBrain.bl', 'Ventricles.bl', 'Hippocampus.bl', 'MidTemp.bl', 'Entorhinal.bl']

and the target variable is:

['MMSE.bl']


1 - Execute `{FEDBIOMED_DIR}/scripts/fedbiomed_run node --config config-n1.ini dataset add` and select CSV data type
2 - Give a dataset name, and tags: `adni`
3 - Select the CSV file

Please apply same operation for other nodes.


## Starting Nodes

The nodes can be started before or after the dataset is added. Please run following command to start a node.

```
{FEDBIOMED_DIR}/scripts/fedbioden_rum node --config <config-file-name> start
```

If no `config` option provided it will start a default node.




