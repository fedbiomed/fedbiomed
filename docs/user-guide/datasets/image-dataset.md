# Image Datasets

## Introduction

`ImageFolderDataset` handles image classification tasks where images are organized in class-based folders. It provides seamless integration with standard computer vision workflows and automatic format conversion for federated learning.

**Key Features:**
- Automatic class discovery from folder names
- Supports all common image formats (JPEG, PNG, BMP, TIFF)
- Auto-conversion to PyTorch tensors or NumPy arrays
- Built-in torchvision transformation support

## Folder Structure

Images must be organized with each class in its own subfolder:

```
root/
├── class_1/
│   ├── image1.jpg
│   └── ...
├── class_2/
│   ├── imageA.jpg
│   └── ...
└── class_3/
    └── ...
```

!!! warning "Critical: No Images in Root Directory"
    All images must be placed inside class subdirectories. The root directory should contain **only** class folders, no image files. Placing images directly in the root will cause dataset loading to fail.

**Supported formats**: JPEG, PNG, BMP, TIFF, and other PIL-supported formats.

### Examples

**Animal Classification:**
```
animals/
├── cats/
├── dogs/
└── birds/
```

**Medical Images:**
```
medical_images/
├── normal/
├── abnormal/
└── uncertain/
```

## Data Preparation

Follow these steps to prepare your image dataset for Fed-BioMed:

1. **Create root directory**
   - This directory will contain only class subdirectories (no images)
   
2. **Create class subdirectories**
   - Use descriptive, consistent names (e.g., `normal`, `abnormal`, `cats`, `dogs`)
   - Folder names become class labels and are case-sensitive
   - Avoid spaces and special characters in folder names
   - Classes are sorted alphabetically during loading
   
3. **Place images in class folders**
   - All images must be inside their respective class folders
   - Do NOT place any images directly in the root directory
   - Ensure no nested subdirectories within class folders
   
4. **Validate structure**
   
   Verify your folder structure. Expected output:
   ```
   /path/to/your/images/
   ├── class_1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── class_2/
   │   ├── imageA.png
   │   └── ...
   └── class_3/
       └── ...
   ```
   
5. **Verify data quality**
   - Check for corrupted images
   - Ensure consistent file formats within your dataset
   - Verify image files have correct extensions
   - Remove or relocate mislabeled images

## Deployment

### Node-Side

Register using the Fed-BioMed node CLI:

```bash
fedbiomed node dataset add
# 1. Select "images"
# 2. Path: /path/to/your/images/
# 3. Unique tags and description
```

See [Deploying Datasets](../nodes/deploying-datasets.md) for details.

### Researcher-Side

Access image datasets through experiment configuration:

```python
from fedbiomed.researcher.experiment import Experiment

# Select nodes with image classification datasets
experiment = Experiment(
    tags=['#images', '#classification', '#medical'],
    model=my_cnn_model,
    training_plan_class=ImageClassificationPlan,
    training_args=training_args
)
```

## Transformations

Image datasets in Fed-BioMed support the full range of torchvision transformations for preprocessing and augmentation.

### Example Medical Image Analysis
```python
# Conservative augmentation for medical images
medical_transform = transforms.Compose([
    transforms.Resize((512, 512)),      # High resolution for medical detail
    transforms.CenterCrop(512),         # Preserve central anatomy
    transforms.RandomRotation(5),       # Minimal rotation
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```
### Framework-Specific Considerations

**For PyTorch Models:**
- Use `transforms.Normalize()` with appropriate mean/std values
- Consider model input size requirements

**For Scikit-learn Models:**
- Images are automatically converted to NumPy arrays
- Consider flattening transforms for traditional ML algorithms
- Normalize pixel values appropriately (typically 0-1 or -1 to 1)

For comprehensive transformation documentation, see [Applying Transformations](applying-transformations.md).

## Best Practices

### Data Organization
- **Consistent structure**: Maintain identical folder organization across all federated nodes
- **Class naming**: Use descriptive, consistent class names across nodes
- **Quality control**: Remove corrupted or mislabeled images before deployment
- **Data quality standards**: Establish and maintain consistent data quality criteria

### Performance Optimization
- **Appropriate transforms**: Choose transforms based on your specific model requirements
- **Batch size tuning**: Balance memory usage with training efficiency
- **Input size considerations**: Match input dimensions to your model architecture

## Troubleshooting

### Common Issues

**Empty Dataset Error:**
- Verify folder structure has class subdirectories
- Check that image files are inside class folders, not in root
- Ensure image files have supported extensions

**Class Mismatch:**
- Verify consistent class folder names across federated nodes
- Check for case sensitivity issues in folder names

**Memory Issues:**
- Reduce batch size in training configuration
- Consider smaller input image dimensions
- Optimize transforms to reduce memory usage

**Transform Errors:**
- Verify transforms are compatible with your training framework
- Ensure normalize values match your model requirements

## Next Steps

- **Learn about transformations**: See [Applying Transformations](applying-transformations.md) for detailed preprocessing guidance
- **Deployment details**: Read [Deploying Datasets](../nodes/deploying-datasets.md) for comprehensive setup instructions
- **Other dataset types**: Explore [Medical Datasets](medical-folder-dataset.md) for specialized medical imaging or [Custom Datasets](custom-dataset.md) for advanced use cases