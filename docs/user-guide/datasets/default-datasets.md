# Default Datasets

## Introduction

Default datasets in Fed-BioMed are pre-built, ready-to-use datasets with automatic downloading and integration. They're ideal for prototyping, education, and benchmarking federated learning approaches.

**Available Datasets:**
- **MNIST**: Handwritten digit recognition
- **MedNIST**: Medical image classification

## Key Features

- Automatic downloading when not present
- Framework compatibility (PyTorch tensors, NumPy arrays)
- Zero configuration required
- Standardized benchmarks

## Deployment

### Node-Side

Deploy using the Fed-BioMed node CLI:

```bash
fedbiomed node dataset add
# 1. Select "default" for MNIST or "mednist" for MedNIST
# 2. Specify dataset location
# 3. Add unique tags and description
```

!!! warning "Important: Path Configuration"
    The path you specify when adding the dataset must match the `root` directory in the data structures shown below (e.g., the parent directory containing `MNIST/` or `MedNIST/`). If the path doesn't match an existing dataset location, Fed-BioMed will download the dataset again to the specified location.

See [Deploying Datasets](../nodes/deploying-datasets.md) for details.

### Researcher-Side

Access datasets through experiment configuration:

```python
from fedbiomed.researcher.experiment import Experiment

experiment = Experiment(
    tags=['#MNIST', '#dataset'],
    model=my_model,
    training_plan_class=MyTrainingPlan,
    training_args=training_args
)
```

!!! note "Tag Matching"
    The tags specified in the experiment configuration must match the tags assigned when registering the dataset on nodes. Only nodes with datasets that have matching tags will participate in the training. Use descriptive and consistent tags across your federated network to ensure proper dataset selection.

## MNIST Dataset

### Overview

Classic handwritten digit classification dataset.

**Dataset Characteristics:**
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 pixels
- **Classes**: 10 (digits 0-9)
- **Color**: Grayscale (single channel)
- **File format**: IDX format (automatically handled)

### Data Structure

The dataset automatically manages the standard MNIST IDX format:

```
root/
└── MNIST/
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

### Sample Data Format

```python
# For PyTorch TrainingPlan - data comes as torch.Tensor
data.shape   # torch.Size([1, 28, 28]) - single channel, 28x28
data.dtype   # torch.float32
data.min()   # 0.0 (black pixels)
data.max()   # 1.0 (white pixels)
target       # tensor(7) - digit class 0-9
```
```python
# For scikit-learn TrainingPlan - data comes as numpy.ndarray
data.shape   # (28, 28) - can be flattened to (784,)
data.dtype   # uint8
data.min()   # 0 (black pixels)
data.max()   # 255 (white pixels)
target       # array(7) - digit class 0-9
```

### MNIST Transform Examples

**Basic Digit Recognition**
```python
mnist_transform = transforms.Compose([
    transforms.RandomRotation(10),                    # Slight rotation for digits
    transforms.Normalize((0.1307,), (0.3081,))       # MNIST-specific normalization
])
```

**Scikit-learn Compatible**
```python
def mnist_sklearn_transform(data):
    flattened = data.flatten()              # Flatten to 784 features
    normalized = (flattened - 33.328) / 78.565  # MNIST normalization
    return normalized
```

## MedNIST Dataset

### Overview

The MedNIST dataset contains medical images from different imaging modalities, designed specifically for medical AI applications.

**Dataset Characteristics:**
- **Total samples**: 58,954 medical images
- **Classes**: 6 medical imaging modalities
  - AbdomenCT: Abdominal CT scans
  - BreastMRI: Breast MRI images  
  - ChestCT: Chest CT scans
  - CXR: Chest X-Ray images
  - Hand: Hand X-Ray images
  - HeadCT: Head CT scans
- **Color**: RGB (converted from medical imaging formats)
- **File format**: JPEG format (automatically handled)

### Medical Data Structure

```
root/
└── MedNIST/
    ├── AbdomenCT/        # Abdominal CT scans
    │   ├── 000000.jpeg
    │   ├── 000001.jpeg
    │   └── ...
    ├── BreastMRI/        # Breast MRI images
    ├── ChestCT/          # Chest CT scans
    ├── CXR/              # Chest X-Ray images
    ├── Hand/             # Hand X-Ray images
    └── HeadCT/           # Head CT scans
        └── ...
```

### Medical Sample Data Format

```python
# For PyTorch TrainingPlan - data comes as torch.Tensor
data.shape   # torch.Size([3, 64, 64])
data.dtype   # torch.float32
target       # tensor(3) - medical modality class 0-5

# For scikit-learn TrainingPlan - data comes as numpy.ndarray
data.shape   # (64, 64, 3)
data.dtype   # uint8
target       # array(3) - medical modality class 0-5

# Class mapping:
# 0: AbdomenCT, 1: BreastMRI, 2: ChestCT, 3: CXR, 4: Hand, 5: HeadCT
```

### Medical Transform Examples

**Conservative Medical Augmentation**
```python
medical_transform = transforms.Compose([
    transforms.Resize((224, 224)),                    # Standard medical image size
    transforms.RandomRotation(5),                     # Conservative rotation
    transforms.ColorJitter(brightness=0.1),           # Slight brightness adjustment
    transforms.Normalize(mean=[0.5], std=[0.5])       # Medical image normalization
])
```

**Medical ML Pipeline (Scikit-learn)**
```python
def medical_sklearn_transform(data):
    resized = cv2.resize(data, (32, 32))             # Smaller for ML algorithms
    flattened = resized.flatten()                    # Flatten for traditional ML
    normalized = (flattened - 127.5) / 127.5         # Normalize to [-1, 1]
    return normalized
```

## Best Practices

### For Node Administrators
- Use descriptive and consistent tags when registering default datasets
- Ensure adequate storage space for automatic downloads

### For Researchers
- Start with default datasets for initial federated learning experiments
- Use consistent preprocessing across experiments for fair comparison
- Leverage default datasets for validating new federated learning algorithms

### For Educational Use
- Begin with MNIST for understanding basic federated learning concepts
- Progress to MedNIST for understanding domain-specific challenges
- Use established transforms and benchmarks for learning

## Next Steps

- **Learn about transformations**: See [Applying Transformations](applying-transformations.md) for preprocessing and augmentation
- **Deployment details**: Read [Deploying Datasets](../nodes/deploying-datasets.md) for comprehensive deployment instructions
- **Advanced datasets**: Explore [Image Datasets](image-dataset.md), [Medical Datasets](medical-folder-dataset.md), or [Custom Datasets](custom-dataset.md) for specialized use cases