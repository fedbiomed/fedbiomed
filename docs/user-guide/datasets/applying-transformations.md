# Applying Transformations

## Introduction

Transformations are custom functions that preprocess your data before training. You can use them to normalize images, scale tabular features, perform data augmentation, and many other preprocessing tasks. Transformations are defined in the `TrainingPlan` as dataset parameters, and Fed-BioMed automatically applies them on each node during data loading. This guide explains how to write transform functions that work with different data types and machine learning frameworks.

## Transform Pipeline Overview

Fed-BioMed's transformation pipeline consists of distinct stages, with **user transforms** being the primary customization point:

```
Raw Data → Node Processing → Framework Conversion → 🎯 User Transforms → ML Framework
    │            │                   │                    │                │
Files/DB    Standardized       Framework-Ready        Customized         Training
            Format             Types                  Processing         Ready
```

!!! note "Key Points"
    - **What you control**: Custom preprocessing, augmentation, normalization logic, etc.
    - **What Fed-BioMed handles**: Data loading and framework conversion.
    - **Your responsibility**: Ensure transforms respect framework type requirements:
        - **PyTorch**: Input/output `torch.Tensor` 
        - **Scikit-learn**: Input/output `numpy.ndarray`

## Data Types and Transform Requirements

This section details what data the node provides and what your transforms should expect as input/output for each framework.

**Node Data → Framework Conversion → Transform Input**

The table below illustrates Fed-BioMed's automatic data conversion pipeline. **Before your custom transform function is called**, Fed-BioMed automatically converts the node-provided data into framework-appropriate types. Your transform function then receives this pre-converted data as input.

| Dataset Type | Node Data Format | **Fed-BioMed converts to →** PyTorch Input | **Fed-BioMed converts to →** Scikit-learn Input |
|--------------|------------------|---------------------------------------------|--------------------------------------------------|
| **Tabular** | `polars.DataFrame` | `torch.Tensor` | `numpy.ndarray` |
| **Medical** | `dict` of `nibabel.Nifti1Image` + `dict` (demographics) | `Dict[str, torch.Tensor]` | `Dict[str, numpy.ndarray]` |  
| **Image** | `PIL.Image.Image` | `torch.Tensor (C, H, W) [0-1]` | `numpy.ndarray (H, W, C) [0-255]` |
| **Native/Custom** | Variable | `torch.Tensor` (best-effort conversion) | `numpy.ndarray` (best-effort conversion) |

!!! note "Understanding the conversion process"
    - **Node Data Format**: The data format provided by Fed-BioMed nodes after loading and initial processing
    - **Fed-BioMed Auto-Conversion**: Fed-BioMed automatically applies framework-specific conversions before calling your transform (e.g., `DataFrame.to_torch().float()` for PyTorch, `DataFrame.to_numpy()` for scikit-learn)  
    - **Your Transform Input**: The converted data type that your custom transform function will receive and must handle

**Transform Output Requirements**

The output format of your custom transform function is **crucial** and differs significantly between frameworks. Understanding these requirements is essential for successful federated learning implementation.

!!! warning "Framework Output Requirements"
    - **PyTorch**: Can handle flexible output structures (`torch.Tensor` or `Dict[str, torch.Tensor]`) because the model's method `training_step` can organize data as needed.
    - **Scikit-learn**: Requires **flattened arrays only** (`numpy.ndarray`) - the exact format the ML model will consume. No dictionaries or complex structures are allowed.

## Common Transform Operations

!!! note "Key Constraint"
    Transforms operate on individual samples without access to global dataset statistics. Use pre-defined constants from domain knowledge instead of computed statistics.

**General Patterns:**

- **Normalization**: Use known constants like `(data - known_mean) / known_std` or simple scaling `data / max_value`
- **Scaling**: Apply min-max with known ranges `(data - min_val) / (max_val - min_val)` or standardization to 0-1
- **Augmentation**: Use sample-level transforms like rotations, flips, noise addition with fixed parameters
- **Feature Engineering**: Create polynomial features, one-hot encoding, or domain-specific transformations
- **Clipping**: Apply bounds with `torch.clamp()` or `np.clip()` using known value ranges

**Data-Type Specific Examples:**

- **Tabular**: Feature scaling, categorical encoding, log transforms for count data
- **Images**: Resizing, intensity normalization (e.g., ImageNet constants), geometric augmentation  
- **Medical**: Protocol-specific normalization, conservative augmentation preserving clinical features, multi-modal combination

## Writing Transform Functions

This section provides complete code examples showing how to write transform functions for different data types. Each example demonstrates the input types you'll receive, the processing you can apply, and the output formats required by each framework.

**Tabular Data Transforms:**

Tabular transforms handle individual samples from structured datasets. Each transform receives a single row of features as input.

```python
# PyTorch tabular transform
def tabular_pytorch_transform(data: torch.Tensor) -> torch.Tensor:
    # Input: torch.Tensor (float32) - numerical/categorical features
    # Output: torch.Tensor
    processed_features = preprocess(data)
    
    # Simple tensor output
    return processed_features

# Scikit-learn tabular transform 
def tabular_sklearn_transform(data: numpy.ndarray) -> numpy.ndarray:
    # Input: numpy.ndarray (float64) - numerical/categorical features
    # Output: numpy.ndarray (MUST be flattened, ready for sklearn model)
    processed = preprocess(data)
    return processed.flatten() if processed.ndim > 1 else processed
```

**Medical Image Transforms:**

Medical transforms work with multi-modal data combining imaging (MRI, CT) and demographic information.

```python
# PyTorch medical transforms - ALWAYS receive Dict[str, torch.Tensor]
# OUTPUT OPTIONS: Multiple formats supported
def medical_pytorch_transform(data: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # Input - medical data modalities
    
    # OPTION 1: Preserve original modality structure
    def option1_preserve_modalities():
        processed = {}
        for modality_name, tensor in data.items():
            if modality_name == 'demographics':
                processed[modality_name] = process_demographics(tensor)
            else:
                processed[modality_name] = apply_medical_preprocessing(tensor)
        return processed  # Dict[str, torch.Tensor]
    
    # OPTION 2: Return single tensor (any dimensions supported)
    def option2_single_tensor():
        # flattened and combined
        all_features = []
        for modality_name, tensor in data.items():
            processed = apply_medical_preprocessing(tensor)
            all_features.append(processed.flatten())
        return torch.cat(all_features)  # [combined_features]
        
        # stacked modalities  
        imaging_tensors = [data['T1'], data['T2']]
        return torch.stack(imaging_tensors, dim=0)  # [2, C, H, W]
    
    # Choose any option - train_step handles all formats
    return option1_preserve_modalities()  # option2_single_tensor()

# Scikit-learn medical transform - receives Dict but MUST return flattened array
def medical_sklearn_transform(data: Dict[str, numpy.ndarray]) -> numpy.ndarray:
    # Input: Dict[str, numpy.ndarray] - medical data modalities
    # Output: numpy.ndarray (MUST be single flattened 1D array combining all modalities)
    
    all_features = []
    for modality_name, array in data.items():
        if modality_name == 'demographics':
            # Handle demographic data
            demo_features = process_demographics(array).flatten()
            all_features.append(demo_features)
        else:
            # Handle imaging modalities - flatten each
            img_features = apply_medical_preprocessing(array).flatten()
            all_features.append(img_features)
    
    # ONLY option: single flattened array
    return numpy.concatenate(all_features)  # [all_features_combined]
```

**Image Transforms:**

Standard image transforms handle computer vision data, from PIL images converted to either tensors or arrays and with different value ranges and shapes in each case.

```python
# PyTorch image transforms
def image_pytorch_transform(data: torch.Tensor) -> torch.Tensor:
    # Input: torch.Tensor [C, H, W] values 0-1 - PIL converted to tensor
    # Output: torch.Tensor
    processed = apply_image_preprocessing(data)
    
    # Simple tensor or complex dictionary both work
    return processed  # or return {"image": processed, "augmented": augmented_version}

# Scikit-learn image transforms
def image_sklearn_transform(data: numpy.ndarray) -> numpy.ndarray:
    # Input: numpy.ndarray [H, W, C] values 0-255 - PIL as array
    # Output: numpy.ndarray (MUST be flattened 1D array for sklearn model)
    processed = apply_image_preprocessing(data)
    return processed.flatten()  # Always flatten for sklearn
```

**Native/Custom Dataset Transforms:**

Native transforms handle custom data formats that don't fit standard categories. Fed-BioMed attempts automatic conversion but may need manual handling.

```python
def flexible_pytorch_transform(data: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # Input: Variable type (depends on source dataset)
    # Output: torch.Tensor OR Dict[str, torch.Tensor] (train_step handles both)
    processed = convert_and_process(data)
    # Can return complex structures
    return {"processed": processed, "metadata": extract_metadata(data)}

def flexible_sklearn_transform(data: Any) -> numpy.ndarray:
    # Input: Variable type (depends on source dataset)  
    # Output: numpy.ndarray (MUST be flattened 1D array)
    processed = convert_and_process(data)
    return numpy.array(processed).flatten()  # Always flatten for sklearn
```

## Using Transforms in Training Plans

Transforms are applied through dataset configuration within Training Plans. The dataset's parameters `transform` and `target_transform` accept your custom transformation functions, which Fed-BioMed calls automatically during data loading.

**Training Plan Setup:**
```python
def my_custom_transform(data):
    # Fed-BioMed ensures input matches framework expectations
    # Apply your preprocessing logic
    return processed_data  # Must match framework output requirements
...
# In your Training Plan
def training_data(self):
    dataset = MyDataset(
        transform=my_custom_transform,  # Your transform function
        target_transform=my_target_transform  # Optional target transform
    )
    return DataLoader(dataset, batch_size=32)
```

!!! note "Key Points"
    - **One transform per data item**: 'data' and 'target' handle independent transforms
    - **Automatic execution**: Fed-BioMed calls your transforms on each data sample
    - **Framework consistency**: Same transform works across all nodes using the same framework
    - **Validation**: Fed-BioMed validates transform outputs automatically

## Validation and Type Safety

!!! warning "Automatic Validation"
    Fed-BioMed automatically validates transform inputs/outputs. Focus on following framework requirements and avoiding common issues below.

**Key Considerations**

- PyTorch Transforms:
    - Input: `torch.Tensor` or `Dict[str, torch.Tensor]`
    - Output: Must return `torch.Tensor` or `Dict[str, torch.Tensor]` 
    - Ensure `float32` dtype for features

- Scikit-learn Transforms:
    - Input: `numpy.ndarray` or `Dict[str, numpy.ndarray]`
    - Output: Must return **flattened** `numpy.ndarray` only
    - Use `.flatten()` or `.reshape(-1)` for sklearn compatibility

## Best Practices

- **Type consistency**: Always return the expected framework type (tensor for PyTorch, array for sklearn)
- **Shape awareness**: For sklearn, always flatten final output
- **Dtype preservation**: Maintain appropriate data types (specific numeric types are expected by some models)
- **Error resilience**: Design transforms to handle edge cases gracefully without crashing training
