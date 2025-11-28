# Datasets in Fed-BioMed

## Introduction

Dataset classes in Fed-BioMed bridge raw data stored on nodes with the federated learning training process. They provide a standardized interface for data access across different data types and machine learning frameworks while maintaining data privacy.

## Purpose

In federated learning, training occurs across multiple distributed nodes, each with potentially different data types and structures. Dataset classes solve the fundamental challenge of **unified data access in heterogeneous environments**.

**Challenge:**

- Data is distributed across nodes with varying formats (images, CSV, NIfTI, etc.)
- Different ML frameworks require different data formats (PyTorch tensors vs. NumPy arrays)
- Raw data must remain private and never leave nodes
- Training code needs consistent data interfaces across all nodes

**Fed-BioMed Solution:**

Dataset classes provide an **abstraction layer** that:

- Standardizes data loading regardless of underlying format
- Automatically converts data to the required framework format
- Enables preprocessing and augmentation at the data source
- Maintains a consistent interface for training code across the federation

**Benefit:**

Researchers can write training plans once and deploy them across nodes with diverse data sources, while node administrators maintain full control over their local data without exposing raw files.

## Key Features

- **Standardized access**: Unified interface for images, tabular, and medical data
- **Framework-agnostic**: Automatic conversion to PyTorch (`torch.Tensor`) or scikit-learn (`numpy.ndarray`) formats
- **Privacy preserving**: Data remains local on nodes
- **Transformation support**: Apply preprocessing and augmentation
- **Cross-node consistency**: Harmonized data formats across federated networks

## Core Elements

All Fed-BioMed datasets share these core concepts:

1. **Node registration**: Nodes deploy datasets with **unique** tags
2. **Researcher selection**: Researchers can list and select datasets using tags
3. **Automatic resolution**: Fed-BioMed nodes resolve tags to local paths
4. **Format conversion**: Data converted to appropriate framework format
5. **Custom transformations**: Flexible transformations supported for data preprocessing and augmentation.

## Using Datasets

### Deploying Datasets on Nodes

Before use in federated training, nodes must [deploy datasets](../nodes/deploying-datasets.md) with unique tags. This registers metadata and makes datasets discoverable by researchers.

Use the following command to add a dataset into the node located in the directory `./my-node`

``` shell
$ fedbiomed node --path my-node dataset add
```

### Searching for Available Datasets

Researchers can identify available by [searching with tags](../researcher/listing-datasets-and-selecting-nodes.md):

```python
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.config import config

req = Requests(config=config)
result = req.list()
```

The search result returns a dictionary mapping nodes to their datasets.

### In Federated Training

Datasets are then referenced by tags in [experiment configuration](../researcher/experiment.md):

```python
# Researcher side - select datasets by tags
experiment = Experiment(
    tags=['#MNIST', '#dataset'],
    model=model,
    training_plan_class=MyTrainingPlan,
    training_args=training_args
)
```

### Local Testing

Datasets can be used for model testing and development when there is data available locally.

## Dataset Types

Fed-BioMed supports several dataset types for different data modalities.

- [Default Datasets](default-datasets.md): Pre-built datasets with automatic downloading (MNIST, MedNIST)
- [Image Datasets](image-dataset.md): Image classification with folder-based organization (`ImageFolderDataset`)
- [Tabular Datasets](tabular-dataset.md): Structured data in CSV format with numerical and categorical features
- [Medical Datasets](medical-folder-dataset.md): Multi-modal medical imaging in NIfTI format with optional demographics (`MedicalFolderDataset`)
- [Custom Datasets](custom-dataset.md): Specialized data types (`CustomDataset`, `NativeDataset`)
