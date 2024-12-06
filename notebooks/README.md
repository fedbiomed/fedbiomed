# Example Notebooks

This directory contains example notebooks for testing purposes oriented to contributors, developers or users who has already some knowledge regarding how Fed-BioMed components works. If you are a beginner please visit our documentation site https://fedbiomed.org for more detailed examples or go to the folder `tutorials` that points to tutorials published in the documentation site. The notebooks in the main directory do not contain detailed information, however, you can find more detailed examples in the `tutorials` directory.

## Datasets Used in the examples

Examples in the main directory use different dataset types. While some datasets are shared across the examples, others are generated within the example notebooks. The instructions for loading generated datasets are available in the notebooks.

### Loading MNIST Dataset

1. `fedbiomed node dataset add`
  * Select option 2 (default) to add MNIST to the node.
  * Confirm default tags by hitting "y" and ENTER.
  * Pick the folder where MNIST is downloaded.
  * The dataset should now be added (if you get a warning saying that data must be unique is because it's been already added)

2. Check that your data has been added by executing `fedbiomed node dataset list`
3. Run the node using `fedbiomed node start`. Wait until you see the output `Starting task manager`: it means the node is active and ready to execute commands.
4. Following the same procedure, you can create a second node for client 2.


### Loading Pseudo Adni Dataset

1. `fedbiomed node dataset add`
  * Select option 1 to add a csv file to the node
  * Choose the name, tags and description of the dataset
    * use `#test_data` for the tags
  * Pick the .csv file from your PC, located under `fedbiomed/notebooks/data/CSV` folder (here: [pseudo_adni_mod.csv](./data/CSV/pseudo_adni_mod.csv))
  * The dataset should now be added
2. Check that your data has been added by executing `fedbiomed node dataset list`
3. Run the node using `fedbiomed node start`. Wait until you see the output `Starting task manager`: it means the node is active and ready to execute commands.

Note: Notebooks/examples may use different tags. Please use the tags that are relevant to the example you are running.

### Loading Adni Dataset

Note: This dataset is different from "Pseudo Adni Dataset". It contains three different CSV files corresponding to three nodes.

This dataset represents a realistic dataset of (synthetic) medical information mimicking the ADNI dataset (http://adni.loni.usc.edu/). The data is entirely synthetic and randomly sampled to mimick the variability of the real ADNI dataset**. The training partitions are available at the following link:

https://drive.google.com/file/d/1R39Ir60oQi8ZnmHoPz5CoGCrVIglcO9l/view?usp=sharing

The federated task we aim to solve is to predict a clinical variable (the mini-mental state examination, MMSE) from a combination of demographic and imaging features. The regressor variables are the following features:

['SEX', 'AGE', 'PTEDUCAT', 'WholeBrain.bl', 'Ventricles.bl', 'Hippocampus.bl', 'MidTemp.bl', 'Entorhinal.bl']

and the target variable is:

['MMSE.bl']


1 - Execute `fedbiomed node -d fbm-node dataset add` and select CSV data type
2 - Give a dataset name, and tags: `adni`
3 - Select the CSV file

Please apply same operation for other nodes.


## Starting Nodes

The nodes can be started before or after the dataset is added. Please run the following command to start a node.

```
fedbioden node --d <component-directory> start
```

If no `config` option provided it will start a default node.




