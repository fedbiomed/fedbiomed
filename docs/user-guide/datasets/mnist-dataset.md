# Using `MnistDataset` in Federated Workflows

`MnistDataset` is a specialized pre-built dataset class designed specifically for the MNIST handwritten digit recognition dataset. This dataset class provides seamless access to the classic MNIST dataset with automatic downloading, preprocessing, and integration with Fed-BioMed's federated learning workflows. 

### MNIST Details

- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 pixels
- **Classes**: 10 (digits 0-9)
- **Color**: Grayscale
- **File format**: IDX format (automatically handled)
  
### Key Features of `MnistDataset`

- **MNIST-specific:** Designed exclusively for the MNIST handwritten digit dataset. Downloads MNIST data automatically if not present.
- **Framework compatibility:** Automatic format conversion for PyTorch (`torch.Tensor`) and scikit-learn (`numpy.ndarray`) - data format matches your training plan type.

## Considerations from Node side

### Data Structure
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

#### Node-Side MNIST Setup (Dataset registration)
```bash
# Nodes handle MNIST data registration and downloads
fedbiomed node dataset add

# Select option 'default' for MNIST dataset (automatic download)
# Use default tags: #MNIST #dataset
```

**Node responsibilities:**
- Download and store MNIST data locally
- Register dataset with unique tags for researcher discovery
- Handle IDX file format and local file system access
- Serve MNIST data samples during federated training

## Considerations from Researcher side

### Sample Data Structure

When you access a sample using `dataset[index]` without applying transformations, you get a tuple `(data, target)` where:

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

### Transform Examples

**Data Augmentation for MNIST using torchvision.transforms**

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

**Custom Scikit-learn Transforms**

```python
# For scikit-learn TrainingPlan - working with numpy arrays
def custom_sklearn_transform(data):
    # data is numpy.ndarray with shape [28, 28]
    flattened = data.flatten()  # Flatten to 784 features
    normalized = (flattened - 33.328) / 78.565
    return normalized

dataset = MnistDataset(transform=custom_sklearn_transform)
```

## Error Handling & Validation

The dataset provides comprehensive error handling:

- **Automatic download:** Handles network issues and corrupted downloads
- **Data validation:** Ensures MNIST data integrity
- **Transform validation:** Validates transform compatibility
- **Format consistency:** Ensures proper data format conversion

## Key Points

- **Nodes handle MNIST data storage and access** (downloads, file paths, data serving) while **researchers define how to process MNIST data** (transforms, normalization, augmentation). Researchers work with the concept of "MNIST" without needing to know where the actual files are stored.
- **Automatic Format Conversion** The dataset automatically converts MNIST data to the required format **based on your training plan type**:
   - *PyTorch TrainingPlan*: Provides `torch.Tensor` with shape [1, 28, 28], values normalized on range [0, 1]
  - *scikit-learn TrainingPlan*: Provides `numpy.ndarray` with shape [28, 28], values on range [0, 255]
