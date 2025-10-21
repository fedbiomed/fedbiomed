# Using `SimpleDataset` Classes in Federated Workflows

## Introduction

Fed-BioMed provides several pre-built dataset classes that inherit from a common `SimpleDataset` base class. These classes are designed for specific, well-defined data formats and provide automatic data loading and preprocessing capabilities.

**Important**: The `SimpleDataset` base class **cannot be instantiated directly**. You must use one of its concrete subclasses that implement specific data loading logic through controllers.

The `SimpleDataset` classes available are:
- `ImageFolderDataset` - For image classification datasets organized in folders
- `MnistDataset` - For MNIST dataset (digit recognition )
- `MedNistDataset` - For MedNIST dataset (medical imaging)

### Sample Data Structure

`SimpleDataset` classes are designed for **image classification tasks** where:
- **Data**: Contains image data (PIL Image, numpy array, or similar image format)
- **Target**: Contains a numeric class label (integer representing the class index)

When you access a sample using `dataset[index]`, you get a tuple `(data, target)` where `data` is the image and `target` is the corresponding class number.

## Principles

- **Concrete classes only**: The base `SimpleDataset` class cannot be instantiated directly - you must use specific subclasses like `ImageFolderDataset`, `MnistDataset`, or `MedNistDataset`
- `SimpleDataset` classes use **controllers** to handle data loading and sample retrieval automatically
- Data and target formats are **implicitly predefined** by the controller - you don't need to implement data loading logic
- **Automatic format conversion** is provided for both PyTorch (`torch.Tensor`) and scikit-learn (`numpy.ndarray`) formats
- **Transform and target_transform** parameters are supported for data preprocessing
- The classes handle **validation** of transforms and data format compatibility automatically

## Key Features

### Automatic Format Conversion
`SimpleDataset` classes automatically convert data to the required format based on your training framework:
- **PyTorch format**: Converts to `torch.Tensor` using `torchvision.transforms.ToTensor()` for images
- **scikit-learn format**: Converts to `numpy.ndarray`

### Transform Support
`SimpleDataset` classes accept `transform` and `target_transform` parameters for data preprocessing:
- `transform`: Applied to input data (images)
- `target_transform`: Applied to target labels (class numbers)

Transforms must match the training framework and must accept and return the same format type. The framework automatically converts raw data to the training format **before** applying transforms.
  
## Available SimpleDataset Classes

### ImageFolderDataset

For image classification datasets organized in the standard folder structure:

```
root/
├── class_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class_2/
│   ├── image3.jpg
│   └── ...
└── ...
```

### MnistDataset

For MNIST digit recognition datasets in the standard format:

```
root/
└── MNIST/
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

### MedNistDataset

For MedNIST medical imaging datasets:

```
root/
└── MedNIST/
    ├── AbdomenCT/
    │   ├── 000000.jpeg
    │   └── ...
    ├── BreastMRI/
    ├── ChestCT/
    ├── CXR/
    ├── Hand/
    └── HeadCT/
```

## Transform Requirements
- **Format consistency**: Transforms must work with the specified Training Plan (framework compatibility).
- **Input/Output types**: For TORCH format, transforms receive and return `torch.Tensor`; for SKLEARN format, they work with `numpy.ndarray`
- **Preprocessing order**: Raw data → format conversion → transform application

### Using Transforms

**Important**: Transforms must be compatible with your chosen Training Plan. The framework converts raw data to the target format first, then applies your transforms.

```python
from torchvision import transforms
from fedbiomed.common.dataset import ImageFolderDataset

# For TORCH format - # Transform receives torch.Tensor, returns torch.Tensor
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224), 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# Create dataset with transforms
dataset = ImageFolderDataset(transform=data_transform)
```

### Best Practices

1. **Choose the right dataset class**: Use the specific dataset class
2. **Match data format**: Define transforms that matches your training framework
3. **Understand sample structure**: Remember that samples return `(image_data, class_number)` tuples

### Error Handling

`SimpleDataset` classes provide comprehensive error handling:

- **Invalid transforms**: Raises if transform is not callable
- **Initialization errors**: Raises if controller cannot be created
- **Format conversion errors**: Raises if data cannot be converted to target format
- **Transform application errors**: Raises if transforms fail during execution