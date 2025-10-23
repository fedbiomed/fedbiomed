# Datasets in Fed-BioMed

## Introduction

Dataset classes in Fed-BioMed bridge raw data stored on nodes with the federated learning training process. They provide a standardized interface for data access across different data types, storage formats, and machine learning frameworks while maintaining data privacy.

## Purpose of dataset classes in Fed-Biomed

In federated learning, training occurs across multiple distributed nodes, each with potentially different data types, storage formats, and structures. Dataset classes solve the fundamental challenge of **unified data access in heterogeneous environments**.

**The Problem:**
- Data is distributed across nodes with varying formats (images, CSV, NIfTI, etc.)
- Different ML frameworks require different data formats (PyTorch tensors vs. NumPy arrays)
- Raw data must remain private and never leave nodes
- Training code needs consistent data interfaces across all nodes

**The Solution:**
Dataset classes provide an **abstraction layer** that:
- Standardizes data loading regardless of underlying format
- Automatically converts data to the required framework format
- Enables preprocessing and augmentation at the data source
- Maintains a consistent interface for training code across the federation

**The Benefit:**
Researchers can write training plans once and deploy them across nodes with diverse data sources, while node administrators maintain full control over their local data without exposing raw files.

## Key Features

- **Standardized access**: Unified interface for images, tabular, and medical data
- **Framework agnostic**: Automatic conversion to PyTorch (`torch.Tensor`) or scikit-learn (`numpy.ndarray`) formats
- **Privacy preserving**: Data remains local on nodes
- **Transformation support**: Apply preprocessing and augmentation
- **Cross-node consistency**: Harmonized data formats across federated networks

## Dataset Types

Fed-BioMed supports several dataset types for different data modalities:

- **Default Datasets**: Pre-built datasets with automatic downloading (MNIST, MedNIST) - [Learn more →](default-datasets.md)
- **Image Datasets**: Image classification with folder-based organization (`ImageFolderDataset`) - [Learn more →](image-dataset.md)
- **Tabular Datasets**: Structured data in CSV format with numerical and categorical features - [Learn more →](tabular-dataset.md)
- **Medical Datasets**: Multi-modal medical imaging in NIfTI format with optional demographics (`MedicalFolderDataset`) - [Learn more →](medical-folder-dataset.md)
- **Custom Datasets**: Specialized data types (`CustomDataset`, `NativeDataset`) - [Learn more →](custom-dataset.md)

## Common Elements

All Fed-BioMed datasets share these core concepts:

### Data Access Pattern

1. **Node registration**: Nodes register datasets with  **unique** tags
2. **Researcher selection**: Researchers select datasets using tags
3. **Automatic resolution**: Fed-BioMed resolves tags to local paths
4. **Format conversion**: Data converted to appropriate framework format

### Transformations

Datasets support flexible transformation systems for preprocessing and augmentation. Transformations are applied locally on nodes without exposing raw data.

[Learn more about transformations →](applying-transformations.md)

## Using Datasets

### In Federated Training

Datasets are referenced by tags in experiment configuration:

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

Datasets can be used locally for model testing and development when there is data available locally.

## Getting Started

1. **Node administrators**: [Deploy datasets](../nodes/deploying-datasets.md) on nodes
2. **Researchers**: [List and select datasets](../researcher/listing-datasets-and-selecting-nodes.md) in experiments
3. **Data preprocessing**: [Apply transformations](applying-transformations.md)
4. **Custom data**: [Create custom datasets](custom-dataset.md) for specialized needs

## Next Steps

Choose your dataset type:

- [**Default Datasets**](default-datasets.md) - Quick prototyping with MNIST/MedNIST
- [**Image Datasets**](image-dataset.md) - Image classification tasks
- [**Tabular Datasets**](tabular-dataset.md) - Structured data
- [**Medical Datasets**](medical-folder-dataset.md) - Medical imaging
- [**Custom Datasets**](custom-dataset.md) - Specialized data types