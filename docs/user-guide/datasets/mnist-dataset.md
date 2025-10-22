# Using `MnistDataset` in Federated Workflows

## Introduction

`MnistDataset` is a specialized pre-built dataset class designed specifically for the MNIST handwritten digit recognition dataset. This dataset class provides seamless access to the classic MNIST dataset with automatic downloading, preprocessing, and integration with Fed-BioMed's federated learning workflows. 

## Quick Start

For users who want to get started immediately, here's the basic setup. **Note**: Transform choice depends on your training plan framework.

### TorchTrainingPlan (Most Common)

```python
from fedbiomed.common.dataset import MnistDataset
from torchvision import transforms

# For TorchTrainingPlan - data returned comes as torch.Tensor
transform = transforms.Normalize((0.1307,), (0.3081,))
dataset = MnistDataset(transform=transform)
```

### SKLearnTrainingPlan

```python
from fedbiomed.common.dataset import MnistDataset

# For SKLearnTrainingPlan - data returned comes as numpy.ndarray
def normalize_sklearn(data):
    return (data - 0.1307) / 0.3081

dataset = MnistDataset(transform=normalize_sklearn)
```

**Key Points:**
- Automatically downloads MNIST data if not present
- Returns (data, label) tuples in the correct format for your training plan
- No `ToTensor()` needed - data format matches your training plan type

For complete federated learning setup, see the [Integration with Training Plans](#integration-with-training-plans) section.

## Node vs Researcher Responsibilities

In Fed-BioMed's federated architecture, `MnistDataset` involves clear separation of responsibilities:

### Node-Side Responsibilities
- **Data storage and management:** Nodes store MNIST data locally and handle automatic downloads
- **Dataset registration:** Nodes register MNIST datasets with unique tags
- **Local data access:** Nodes manage local MNIST file paths and handle data serving during training
- **Data format handling:** Nodes handle reading and conversion to standard formats

### Researcher-Side Responsibilities  
- **Transform definition:** Researchers define MNIST preprocessing, normalization, and augmentation transforms
- **Model architecture:** Researchers design models compatible with MNIST data
- **Training configuration:** Researchers configure batch sizes, learning rates, and training parameters
- **Task specification:** Researchers specify dataset requirements through experiment configuration

**Nodes handle MNIST data storage and access** (downloads, file paths, data serving) while **researchers define how to process MNIST data** (transforms, normalization, augmentation). Researchers work with the concept of "MNIST" without needing to know where the actual files are stored.

## Key Features

`MnistDataset` provides several key capabilities:

- **MNIST-specific:** Designed exclusively for the MNIST handwritten digit dataset
- **Automatic download:** Downloads MNIST data automatically if not present
- **Framework compatibility:** Automatic format conversion for PyTorch (`torch.Tensor`) and scikit-learn (`numpy.ndarray`)
- **Train/test split:** Supports both training and testing dataset access
- **Fed-BioMed optimized:** Seamless integration with Fed-BioMed federated learning workflows

### Automatic Format Conversion

The dataset automatically converts MNIST data to the required format **based on your training plan type**:
- **PyTorch TrainingPlan**: Provides `torch.Tensor` with shape [1, 28, 28], values normalized on range [0, 1]
- **scikit-learn TrainingPlan**: Provides `numpy.ndarray` with shape [28, 28], values on range [0, 255]

## Transform System

**Transform compatibility depends on your training plan type**. PyTorch transforms expect tensors, while scikit-learn transforms expect numpy arrays.

### Data Augmentation for MNIST

```python
# For PyTorch TrainingPlan - augmentation
augmented_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = MnistDataset(transform=augmented_transform)

# Note: For scikit-learn TrainingPlan, you'd need numpy-based augmentation functions
```

### Custom Scikit-learn Transforms

```python
# For scikit-learn TrainingPlan - working with numpy arrays
def custom_sklearn_transform(data):
    # data is numpy.ndarray with shape [28, 28]
    flattened = data.flatten()  # Flatten to 784 features
    normalized = (flattened - 33.328) / 78.565
    return normalized

dataset = MnistDataset(transform=custom_sklearn_transform)
```

## Data Structure

### Expected Data Structure

The dataset automatically manages the standard MNIST data structure:

```
root/
└── MNIST/
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

### Sample Data Structure

When you access a sample using `dataset[index]`, you get a tuple `(data, target)` where:

- **Data**: image already converted to the target format (tensor for PyTorch, array for scikit-learn)
- **Target**: Integer class label (0-9) representing the digit

Example sample structure:
```python
# For PyTorch TrainingPlan - data already comes as torch.Tensor
data
>> tensor([[[0.0, 0.1, 0.2, ...], ...]])
target
>> tensor(7)

# For scikit-learn TrainingPlan - data already comes as numpy.ndarray
data
>> array([[0, 0, 0, 128, 255], ...], dtype=uint8)
target
>> array(7)
```

## Integration with Training Plans

### TorchTrainingPlan Example

This example shows the key components for using MnistDataset in Fed-BioMed training plans:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MnistDataset
from torchvision import transforms

class MyTrainingPlan(TorchTrainingPlan):
    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    def init_model(self, model_args):
        return self.Net(model_args = model_args)

    def init_optimizer(self, optimizer_args):
        return Adam(self.model().parameters(), lr=optimizer_args["lr"])
    
    def init_dependencies(self):
        return [
            "from torchvision import transforms",
            "from torch.optim import Adam",
            "from fedbiomed.common.dataset import MnistDataset"
        ]

    def training_data(self):
        # Use PyTorch/torchvision transforms
        transform = transforms.Normalize((0.1307,), (0.3081,))
        
        # For PyTorch TrainingPlan: data comes as torch.Tensor
        dataset = MnistDataset(transform=transform)
        
        return DataManager(dataset=dataset, shuffle=True)

    def training_step(self, data, target):
        output = self.model().forward(data)
        return F.nll_loss(output, target)
```

### SKLearnTrainingPlan Example

For comparison, here's how transforms work with scikit-learn training plans:

```python
from sklearn.linear_model import SGDClassifier
from fedbiomed.common.training_plans import SKLearnTrainingPlan
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import MnistDataset

class MyTrainingPlan(SKLearnTrainingPlan):
    def init_model(self, model_args):
        return SGDClassifier(random_state=42)

    def training_data(self):
        # For scikit-learn TrainingPlan: data comes as numpy.ndarray
        # Use numpy-compatible transforms
        def sklearn_transform(data):
            # data is numpy.ndarray with shape [28, 28]
            flattened = data.flatten()  # Flatten to 784 features
            normalized = (flattened - 0.1307) / 0.3081
            return normalized
        
        # MnistDataset automatically provides numpy.ndarray format
        dataset = MnistDataset(transform=sklearn_transform)
        return DataManager(dataset=dataset, shuffle=True)
```

### Node and Researcher Workflow

Following the Fed-BioMed tutorial pattern, the workflow separates node and researcher responsibilities:

#### Node-Side MNIST Setup (Dataset registration)
```bash
# Nodes handle MNIST data registration and downloads
fedbiomed node dataset add

# Select option 'default' for MNIST dataset (automatic download)
# Use default tags: #MNIST #dataset
```

**Node responsibilities:**
- Download and store MNIST data locally (60K training + 10K test images)
- Register dataset with standard tags for researcher discovery
- Handle IDX file format and local file system access
- Serve MNIST data samples during federated training

#### Researcher-Side Experiment Configuration
```python
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.common.metrics import MetricTypes

# Researcher defines experiment parameters (no data paths needed)
model_args = {}

training_args = {
    'loader_args': {
        'batch_size': 48,
    },
    'optimizer_args': {
        'lr': 1e-3,
    },
    'epochs': 1,             # Researcher-defined training epochs
    'test_ratio': 0.25,      # Researcher-defined validation split
    'test_metric': MetricTypes.F1_SCORE,
    'test_on_global_updates': True,
    'test_on_local_updates': True,
}

# Query MNIST datasets by tags (nodes provide the data)
tags = ['#MNIST', '#dataset']
rounds = 4

# Create experiment - framework connects researcher to node data
exp = Experiment(
    tags=tags,                           # Find datasets by tags
    model_args=model_args,
    training_plan_class=MyTrainingPlan,  # Researcher-defined transforms
    training_args=training_args,
    round_limit=rounds,
    aggregator=FedAverage()
)

exp.run()
```

**Researcher responsibilities:**
- Define MNIST transforms and preprocessing in training plans
- Configure model architecture
- Specify training parameters and experimental setup
- No knowledge of MNIST file locations or download management

## Testing and Evaluation

### Model Testing with MNIST Test Set

Example for PyTorch TrainingPlan evaluation:

```python
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Use MNIST for consistent evaluation
test_dataset = datasets.MNIST('./data', transform=transform, train=False, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load trained federated model
fed_model = exp.training_plan().model()
fed_model.load_state_dict(exp.aggregated_params()[rounds-1]['params'])

# Evaluate model performance
def test_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

accuracy = test_accuracy(fed_model, test_loader)
print(f"Test Accuracy: {accuracy:.2f}%")
```

## Common Use Cases

- **Educational Purposes**: Perfect for learning federated learning concepts
- **Research Baselines**: Standardized setup for reproducible research
- **Algorithm Development and Experimentation**: Custom preprocessing for testing new approaches

## Error Handling & Validation

The dataset provides comprehensive error handling:

- **Automatic download:** Handles network issues and corrupted downloads
- **Data validation:** Ensures MNIST data integrity
- **Transform validation:** Validates transform compatibility
- **Format consistency:** Ensures proper data format conversion

## Best Practices

### Performance Optimization
- **Batch size:** Use appropriate batch sizes (32-64) for MNIST
- **Preprocessing:** Apply normalization for better convergence
- **Memory management:** MNIST is small, so memory is rarely an issue

### Model Development
- **Start simple:** Begin with basic CNNs before complex architectures
- **Validate locally:** Test on single node before federated training
- **Monitor convergence:** MNIST should converge quickly (1-5 epochs)

## Supported Frameworks

- **PyTorch**: Full support with automatic tensor conversion and torchvision transforms compatibility
- **scikit-learn**: Full support with numpy array conversion and numpy-compatible transforms

## MNIST Dataset Details

- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 pixels
- **Classes**: 10 (digits 0-9)
- **Color**: Grayscale
- **File format**: IDX format (automatically handled)

## Summary

`MnistDataset` provides a reliable way to work with MNIST data in Fed-BioMed. It eliminates setup complexity while providing flexibility.

Key advantages:
- **Zero setup:** Automatic download and preprocessing
- **Educational friendly:** Perfect for learning federated learning concepts
- **Research ready:** Standardized setup for reproducible research
- **Framework agnostic:** Works with both PyTorch and scikit-learn

This makes it the perfect starting point for anyone getting started with federated learning, reproducing research results, or developing new federated learning algorithms.