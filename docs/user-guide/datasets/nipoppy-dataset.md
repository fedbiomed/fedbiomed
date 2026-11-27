# Nipoppy Dataset

## Introduction

[Nipoppy](https://nipoppy.readthedocs.io) is a framework for the standardized organization and processing of
neuroimaging-clinical data. Please visit the [Nipoppy documentation](https://nipoppy.readthedocs.io) to find out more.

`NipoppyDataset` gives your training plan access to the tabular data of a Nipoppy study. Specifically, it provides:

- **Phenotypes**: columns of the harmonized phenotypic file (`tabular/harmonized.tsv`), named after their
  [Neurobagel](https://neurobagel.org) terms (e.g. `nb:Age`).
- **Derivatives**: TSV files produced by processing pipelines, such as
  [imaging-derived phenotypes](https://nipoppy.readthedocs.io/en/latest/explanations/idp_spec.html).
- **Session selection**: each sample is one participant at one session, among those listed in the study's
  [manifest](https://nipoppy.readthedocs.io/en/latest/explanations/manifest.html).
- **Framework compatibility**: samples are returned in the format of your training plan (PyTorch or scikit-learn).

!!! warning "Limitations"
    - **Tabular data only**: images (e.g. NIfTI files) are not loaded.
    - **One session per participant** by default: loading fails if a participant has several sessions after
      filtering. See [Longitudinal data](#longitudinal-data) to turn this check off.
    - **Numeric samples only**: encode categorical columns with a [dataframe transform](#transforms).
    - **Whole derivative files**: all columns of a selected file become input features, unless you remove them with a
      [dataframe transform](#transforms).
    - **Not supported**: [Data Loading Plans](medical-folder-dataset.md#data-loading-plans-dlp),
      [federated analytics](federated-analytics.md), and registration through the Node GUI.

## Deploying the dataset

Install Nipoppy 0.4.7 or later in the Fed-BioMed environment of the node:

```shell
pip install "nipoppy>=0.4.7"
```

The node can then register a study by providing its root directory, as described in
[Deploying Datasets](../nodes/deploying-datasets.md):

```shell
fedbiomed node dataset add
# 1. Select "nipoppy"
# 2. Name, unique tags and description
# 3. Path: /path/to/nipoppy_study
```

The study must contain at least `manifest.tsv` and `tabular/harmonized.tsv`. Its layout is described in the
[Nipoppy study specification](https://nipoppy.readthedocs.io/en/latest/explanations/study_spec.html).

!!! warning "Register the dataset again after running a new pipeline"
    When a dataset is registered, the node database saves a copy of the study's available phenotypes and sessions,
    installed pipelines and shape. Researchers only see this copy when they
    [list the datasets](#listing-available-data), so they cannot find new pipeline outputs, phenotypes or sessions
    until it is updated.

    **Every time you run a new pipeline**, or otherwise change the study (e.g. update `tabular/harmonized.tsv`), please
    delete the dataset and register it again with the same path and tags:

    ```shell
    fedbiomed node dataset delete   # select the Nipoppy dataset
    fedbiomed node dataset add      # select "nipoppy", same path and tags as before
    ```

    Researchers select datasets by tags, so their experiments keep working with the new registration.

    Training does not use this copy: the node reads the metadata from the study on disk every time it loads the data.
    A change to the study can therefore affect experiments that are already running. If possible, avoid running
    pipelines while an experiment is using the dataset.

## Listing available data

As a researcher, you can find the phenotypes, sessions and pipelines of each dataset by
[listing the datasets](../researcher/listing-datasets-and-selecting-nodes.md) on the nodes:

```python
from fedbiomed.researcher.config import config
from fedbiomed.researcher.requests import Requests

datasets = Requests(config).list()
print(datasets[node_id][0]['dataset_parameters'])
print(datasets[node_id][0]['shape'])
# > OUTPUT:
# > {'available_phenotypes': ['nb:Sex', 'nb:Age', 'nb:Diagnosis', ...],
# >  'available_sessions': ['01', '02'],
# >  'installed_pipelines': {'bidsification': {},
# >                          'processing': {'freesurfer': ['7.3.2']},
# >                          'extraction': {'fs_stats': ['0.2.1']}}}
# > SHAPE:
# > [299, 6]
```

`available_sessions` lists the session ids that have phenotypic data, sorted alphabetically, which is not
necessarily chronological order (e.g. `'M12'` comes before `'M6'`).

This list shows the study as it was when the node registered the dataset. Pipelines run on the node since then are
only listed after the node registers the dataset again, but your training plan can still use them (see
[Selecting derivatives on each node](#selecting-derivatives-on-each-node)).

!!! note "The listed `shape` is an approximation"
    The `shape` of a dataset counts all rows and all phenotypes, including rows with missing values. 
    The first dimension is an **upper bound** on the number of participants in a training round.
    The second dimension is the number of **phenotype** columns, but does not include derivatives.


## Using the dataset in a training plan

You may create a `NipoppyDataset` in the `training_data` method of your [training plan](../researcher/training-plan.md),
then select the dataset through the [experiment](../researcher/experiment.md) `tags`.

```python
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import NipoppyDataset
from fedbiomed.common.training_plans import FedSGDRegressor


class MyTrainingPlan(FedSGDRegressor):

    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import NipoppyDataset"]

    @staticmethod
    def encode_sex(df):
        df = df.copy()
        df['nb:Sex'] = df['nb:Sex'].map({'snomed:248152002': 0.0, 'snomed:248153007': 1.0})
        return df

    def training_data(self):
        dataset = NipoppyDataset(
            phenotypes=['nb:Sex'],
            derivatives=[
                ('freesurfer', '7.3.2', 'idp/fs_stats-0.2.1/fs7.3.2-aparc.DKTatlas-thickness.tsv')
            ],
            target='nb:Age',
            session_filters='01',
            whole_df_level_transform=MyTrainingPlan.encode_sex,
        )
        return DataManager(dataset=dataset, shuffle=True)
```

Indexing the dataset returns a `(data, target)` tuple of one-dimensional arrays or tensors, where `target` is `None`
if no target is selected.

## Main API

The constructor of `NipoppyDataset` takes the following arguments, all of which are optional:

| Argument | Description |
|---|---|
| `phenotypes` | Phenotype(s) to use as input features. Defaults to all available phenotypes. |
| `derivatives` | Derivatives to use as input features, as `(pipeline, version, path)` tuples. |
| `target` | Column(s) to use as target: phenotypes or columns of the selected derivatives. Defaults to no target. |
| `session_filters` | Session(s) to keep. See [Selecting sessions](#selecting-sessions). |
| `one_session_per_participant` | Whether loading fails if a participant has several sessions. Defaults to `True`. See [Longitudinal data](#longitudinal-data). |
| `drop_na` | Whether to drop rows with missing values. Defaults to `True`. |
| `drop_na_kwargs` | Arguments for `pandas.DataFrame.dropna`. |
| `whole_df_level_transform` | Function applied once to the whole dataframe. |
| `sample_level_transform` | Function applied to the input features of each sample. |
| `sample_level_target_transform` | Function applied to the target of each sample. See [Transforms](#transforms). |
| `derivatives_initializer` | Function selecting derivatives on the node. See [Selecting derivatives on each node](#selecting-derivatives-on-each-node). |

### Selecting phenotypes and derivatives

The input features are all selected columns except those in `target`.

```python
# Input features: phenotypes and derivatives
dataset = NipoppyDataset(phenotypes=['nb:Sex'], derivatives=derivatives, target='nb:Age')
# Input features: derivatives only
dataset = NipoppyDataset(phenotypes=[], derivatives=derivatives, target='nb:Age')
# Target: derivative columns; input features: the phenotypes and the other derivative columns
dataset = NipoppyDataset(
    phenotypes=['nb:Sex', 'nb:Age'],
    derivatives=derivatives,
    target=['lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'],
)
```

A target can be a phenotype, even if it is not listed in `phenotypes`, or a column of one of the selected derivative
files. Loading fails if a target column is not found, or if it appears more than once (e.g. in several derivative
files).

!!! warning "Specify `phenotypes` explicitly"
    With `phenotypes=None`, all phenotypes of the study on each node are used, including any non-numeric ones. Their
    list and order may differ between nodes, and may change if a node updates its study during your experiment.

The `path` of a derivative is relative to `derivatives/<pipeline>/<version>/` in the study. It may include wildcards,
but it must match exactly one file.

### Selecting sessions

The argument `session_filters` can be a session id, a list of session ids, or a list of
`(participant_id, session_id)` tuples. Ids do not include the BIDS prefixes `sub-` and `ses-`.

```python
dataset = NipoppyDataset(session_filters='01')
# or
dataset = NipoppyDataset(session_filters=['01', 'baseline'])
# or
dataset = NipoppyDataset(session_filters=[('001', '01'), ('002', 'baseline')])
```

You can find the session ids of each dataset in `available_sessions` when [listing the datasets](#listing-available-data).

!!! note "Session ids may differ between nodes"
    `training_data` runs on the node, so you can choose the session of each node based on `self.node_id()`.

#### Longitudinal data

By default, loading fails if a participant still has several sessions after `session_filters`. Set
`one_session_per_participant=False` to load all of them, with one sample per participant and session:

```python
dataset = NipoppyDataset(session_filters=['01', '02'], one_session_per_participant=False)
```

!!! warning "Samples are not independent"
    The sessions of a participant are separate samples, split at random between the training and testing sets. The
    same participant can therefore appear in both, and counts such as the number of samples count sessions, not
    participants.

### Missing values

By default, rows with a missing value in any selected column are dropped. Derivatives and phenotypes are combined with
an outer join, so this includes participants that are absent from one of the selected files.

```python
# Only drop rows with a missing target
dataset = NipoppyDataset(drop_na_kwargs={'subset': ['nb:Age']})
```

### Transforms

The data goes through the following steps on the node:

1. The selected columns are read.
2. Session filters and `dropna` are applied.
3. `whole_df_level_transform` is applied to the whole `pandas.DataFrame`.
4. Each retrieved sample is converted to your training plan's format (`numpy.ndarray` or `torch.Tensor`).
5. `sample_level_transform` and `sample_level_target_transform` are applied to the sample.

Use the dataframe transform for operations that need non-numeric values or several samples, such as encoding
categorical columns. Sample-level transforms must return the same type they receive.

Sample-level transforms receive a 1-D array or tensor without column names, so columns are indexed by position. The
target has one element per column of `target`, in the same order, including when there is a single target column
(shape `(1,)`).

```python
# PyTorch: nb:Age is an integer column, and the loss needs a float target
dataset = NipoppyDataset(target='nb:Age', sample_level_target_transform=lambda x: x.float())
# Several target columns: y[0] is nb:Age, y[1] is nb:Sex
dataset = NipoppyDataset(target=['nb:Age', 'nb:Sex'], sample_level_target_transform=lambda y: y[0:1])
```

!!! warning "Dataframe transforms"
    The transform must keep the `(participant_id, session_id)` index and the `target` columns, and must return only
    numeric columns. It also runs on each node independently: an encoder fitted on local data may produce different
    columns on different nodes, which is incompatible with aggregating model parameters.

Please visit [Applying Transformations](applying-transformations.md) for more information.

### Selecting derivatives on each node

Pipeline versions may differ between nodes. With `derivatives_initializer`, you can select derivatives based on what is
installed on each node. The function takes the dataset as its only argument, and runs before the data is read.

```python
class MyTrainingPlan(FedSGDRegressor):

    @staticmethod
    def select_derivatives(dataset):
        pipelines = dataset.get_metadata()['installed_pipelines']
        freesurfer = pipelines['processing']['freesurfer'][0]
        fs_stats = pipelines['extraction']['fs_stats'][0]
        dataset.derivatives = [(
            'freesurfer',
            freesurfer,
            f'idp/fs_stats-{fs_stats}/fs{freesurfer}-aparc.DKTatlas-thickness.tsv',
        )]

    def training_data(self):
        dataset = NipoppyDataset(
            phenotypes=['nb:Age'],
            derivatives_initializer=MyTrainingPlan.select_derivatives,
        )
        return DataManager(dataset=dataset, shuffle=True)
```

The method `get_metadata` returns the `available_phenotypes`, `available_sessions` and `installed_pipelines` read
from the study on disk when the node loads the data, at every round. It includes pipelines run after the dataset was
registered, even if they are not listed yet. It returns an empty dictionary until the data is loaded.

!!! tip "Select the same derivatives at every round"
    If a node runs a new pipeline version during your experiment, an initializer that picks e.g. the latest version
    would switch to different input features between rounds. Prefer selecting a specific version when one is available.

### Loading the data

The node calls `load` for you with the study path, so training plans should not call it. You can call `load` yourself,
for example to test locally:

```python
from fedbiomed.common.dataset_types import DataReturnFormat

dataset = NipoppyDataset(phenotypes=['nb:Age'])
dataset.load(root='/path/to/nipoppy_study', to_format=DataReturnFormat.SKLEARN)

data, target = dataset[0]
```
