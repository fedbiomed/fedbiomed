# Medical Folder Dataset

## Introduction

`MedicalFolderDataset` handles medical imaging data in structured folder hierarchies. It supports multi-modal medical data in NIfTI format (.nii, .nii.gz) with optional demographic/clinical information from CSV files, ideal for medical imaging research in federated environments.

## Key Features

- Structured medical data: `root/subject/modality/file.nii.gz`
- Multi-modal support (T1, T2, FLAIR, DWI, etc.)
- NIfTI format (.nii, .nii.gz)
- Demographics integration from CSV that matches patient folders by 'subject ID column'.
- Data Loading Plans (DLP) for cross-site compatibility
- Framework compatibility (PyTorch and scikit-learn)
- Per-modality and global transformations

## Required Folder Structure

Data must follow this exact structure: `root/subject/modality/file`

```
root/
├── subject-01/
│   ├── T1/
│   │   └── subject-01_T1.nii.gz
│   ├── T2/
│   │   └── subject-01_T2.nii.gz
│   └── FLAIR/
│       └── subject-01_FLAIR.nii.gz
├── subject-02/
│   └── ...
└── demographics.csv (optional)
```

**Structural Requirements:**

- **Root directory**: Contains subject subdirectories and optional demographics file
- **Subject level**: Each folder in root correspond to a potential subject (e.g., `subject-01`, `patient_123`, `P001`)
- **Modality level**: Each modality has its own folder within the subject directory (e.g., `T1`, `T2`). Modality folder names should be identical across all subjects.
- **File level**: Each modality folder contains exactly **one** NIfTI file (`.nii` or `.nii.gz`)

!!! warning "Important"
    Must be exactly 3 levels deep. Files at wrong levels are ignored.

!!! note "Flexibility"
    Use Data Loading Plans (DLP) to map different folder names (like `T1_MPRAGE`, `T1w`) to the same modality. See [Data Loading Plans section](#data-loading-plans-dlp) for more details.

### Data Preparation Guidelines

Follow these steps to prepare your medical imaging dataset for Fed-BioMed:

**Organize folder hierarchy**

- Create a root directory that will contain all subject data
- Each subject must have its own subdirectory
- Within each subject directory, create modality-specific subdirectories

**Place NIfTI files correctly**

- Each modality folder should contain exactly one NIfTI file per subject
- Files must have either `.nii` or `.nii.gz` extension
- Avoid placing files directly in subject folders or the root directory

**Ensure modality consistency**

- Modality names must be identical across subjects (case-sensitive) unless using Data Loading Plans (DLP). 
- If using DLP to map multiple folder names to a single modality, ensure no subject has multiple folders that map to the same final modality.
- Check that each required modality is present for all subjects. Subjects with missing modalities will be ignored.

**Prepare demographics file (optional)**

- Create a CSV file with subject-level information
- There must exist a column that matches your subject folder names
- Include relevant clinical variables and keep sensitive information de-identified and compliant with privacy regulations

## Deployment

### Node-Side

Medical imaging datasets can be registered on nodes using either the command-line interface (CLI) or the graphical user interface (GUI).

**Command-Line Interface**

Register using the Fed-BioMed node CLI:

```bash
fedbiomed node dataset add
# 1. Select "medical-folder"
# 2. Path: /path/to/medical_data/
# 3. a. Demographics file (optional): /path/to/demographics.csv
#    b. Index column (optional): 'colum_subject_id'
# 4. Unique tags and description
```

**Graphical User Interface**

Alternatively, datasets can be added through the Fed-BioMed Node GUI, which provides a user-friendly web interface:

**1. Start the Node GUI**

```bash
fedbiomed node gui start
```
Access the GUI at `http://localhost:8484` (default login: `admin@fedbiomed.gui` / `admin`)

**2. Add Dataset**

- Navigate to the dataset management section
- Select "medical-folder" as the dataset type
- Provide the path to your medical imaging data
- Optionally specify demographics file and index column
- Add descriptive tags and description
- Add DLP (mapping of folders to modalities)
- Save the dataset configuration

**Note**: The Node GUI requires additional dependencies. Install them with:
```bash
pip install fedbiomed[gui]
```
- See [Node GUI documentation](../nodes/node-gui.md) for complete setup and usage details.
- See [Deploying Datasets](../nodes/deploying-datasets.md) for comprehensive deployment instructions.

### Researcher-Side

Access medical imaging datasets through [experiment configuration](../researcher/experiment.md):

```python
from fedbiomed.researcher.experiment import Experiment

# Select nodes with multi-modal brain imaging datasets
experiment = Experiment(
    tags=['#medical', '#neuroimaging', '#brain', '#multimodal'],
    model=my_medical_model,
    training_plan_class=MedicalImagingPlan,
    training_args=training_args
)
```

## Data Loading Plans (DLP)

Data Loading Plans solve the common challenge of varying modality folder naming conventions. They provide a standardized way to map different folder names to unified modality names. DLP operates at the **node level** - each node configures its own mapping based on its local folder structure. When researchers query datasets, they see the standardized modality names (the mapping target names) rather than the original folder names. This abstraction allows researchers to write consistent training plans while nodes handle their specific naming variations automatically.

### Motivation

In medical imaging, different clinical centers, time periods, or protocols often use different naming conventions for the same imaging modalities. For example:

- Site A uses `T1_MPRAGE`, Site B uses `T1w`, Site C uses `T1_weighted`
- Within a single site: older scans use `T1_MPRAGE`, newer scans use `T1w`
- All represent the same T1-weighted MRI modality but have different folder names

### How DLP Works

DLP creates a mapping that translates various folder names to standardized modality names. When you define that `T1_MPRAGE`, `T1w`, and `T1_weighted` all map to `T1`, the system treats them as the same modality type.

This mapping works both **across different nodes** (different clinical sites) and **within a single node** (different acquisitions or time periods at the same site).

**Directory Structure Examples:**

- Scenario 1: Cross-Site Variation (Different Nodes):
```
Site A:                    Site B:
root/                      root/
├── subject-01/            ├── patient-01/
│   ├── T1_MPRAGE/         │   ├── T1w/
│   └── T2_FLAIR/          │   └── T2w/
└── subject-02/...         └── patient-02/...
```

- Scenario 2: Within-Site Variation (Single Node):
```
Single Site with Mixed Naming Conventions:
root/
├── subject-01/        # Scanned with old protocol
│   ├── T1_MPRAGE/
│   └── T2_FLAIR/
├── subject-02/        # Scanned with updated protocol  
│   ├── T1w/
│   └── T2w/
├── subject-03/        # Scanned with new naming convention
│   ├── T1_weighted/
│   └── T2_weighted/
└── subject-04/        # Back to standardized naming
    ├── T1_MPRAGE/
    └── T2_FLAIR/
```

**After DLP Mapping (Unified View):**

Both scenarios result in standardized modality names. Same unified modality name regardless of original folder name:
```
root/
├── subject-01/
│   ├── T1/     # Maps from any T1 variant
│   └── T2/     # Maps from any T2 variant
└── ...
```

### Constraints and Error Conditions

**✅ Valid Scenarios:**

- Subject has `T1_MPRAGE/` folder (maps to `T1`) ✓
- Different subjects use different folder names that map to the same final modality ✓
- Subject has some modalities missing → subject will be **ignored** (not an error) ✓
- Subject has `T1_MPRAGE/` folder but it's empty → folder will be **ignored** (not an error) ✓
- Subject has `T1_MPRAGE/` folder with non-NIfTI files → folder will be **ignored** (not an error) ✓

**❌ Error Scenario:**

- Subject has both `T1_MPRAGE/` and `T1w/` folders (both map to `T1`)
- Each subject must have **exactly one folder** that maps to each final modality name
- One file per subject per final modality
    
**Subject Inclusion Logic**

When using DLP, the system applies the following logic to determine which subjects to include:

- **Complete subjects**: Have all required modalities after mapping → **Included**
- **Incomplete subjects**: Missing one or more required modalities → **Automatically ignored**
- **Ambiguous subjects**: Have multiple folders mapping to the same final modality → **Error (dataset fails to load)**

Example with T1 and T2 requirements:

- Subject-01: has `T1_MPRAGE/` and `T2_FLAIR/` → **Included** (both T1 and T2 available)
- Subject-02: has only `T1w/` → **Ignored** (missing T2)
- Subject-03: has `T1_MPRAGE/` and empty `T2w/` folder → **Ignored** (T2 folder invalid)
- Subject-04: has both `T1_MPRAGE/` and `T1w/` → **Error** (ambiguous T1 mapping)


## Samples Structure

Once your medical imaging dataset is registered, you need to understand how to configure and access the data in your training plans. This section explains how to query available modalities and specify which modalities to use for training and targets.

### Available Modalities

After adding a dataset, you can check what modalities are available, querying the datasets available.

- From the node side:
```bash
fedbiomed node dataset list
```

- From the researcher side:
```python
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.config import config

req = Requests(config)
datasets = req.list(verbose=True)
```

Both commands show all registered datasets with their available modalities. The modalities listed are the names you should use when specifying `data_modalities` and `target_modalities` in your dataset configuration.

**What You Get**

Accessing `dataset[index]` returns `(data, target)`:

```python
data = {
    'T1': torch.Tensor([256, 256, 180]),      # Brain scan data
    'T2': torch.Tensor([256, 256, 180]),      # Brain scan data
    'demographics': {'age': 45, 'gender': 'M'} # Optional patient info
}
target = {'label': torch.Tensor([256, 256, 180])}  # Optional target data
```

**How to Configure Modalities**

`MedicalFolderDataset` allows you to specify which modalities to use for data and which for targets:

```python
from fedbiomed.common.dataset import MedicalFolderDataset

dataset = MedicalFolderDataset(
    data_modalities=['T1', 'T2'],
    target_modalities=['label'],
)

# Use multiple modalities for data, no specific target
dataset = MedicalFolderDataset(
    data_modalities=['T1', 'T2', 'FLAIR', 'DWI'],
    target_modalities=None,
)
```

### Basic Usage Examples

- Single Modality Dataset:
```python
# Simple T1-weighted MRI dataset
dataset = MedicalFolderDataset(
    data_modalities='T1',
    target_modalities=None
)
```

- Multi-Modal Analysis:
```python
# Multi-modal brain tumor segmentation
dataset = MedicalFolderDataset(
    data_modalities=['T1', 'T2', 'demographics'],
    target_modalities=['label'],  # Segmentation mask as target
    transform={
        'T1': lambda x: torch.from_numpy(x.get_fdata()).float(),
        'T1ce': lambda x: torch.from_numpy(x.get_fdata()).float(), 
        'T2': lambda x: torch.from_numpy(x.get_fdata()).float(),
        'FLAIR': lambda x: torch.from_numpy(x.get_fdata()).float()
    },
    target_transform={
        'seg': lambda x: torch.from_numpy(x.get_fdata()).long()
    }
)
```

### Transforms Example

Medical datasets support sophisticated transformation systems tailored for medical imaging workflows:

**Per-Modality Medical Transform**
```python
    # Example inside a TorchTrainingPlan:
    ...
    def training_data(self):
        ...
        def demographics_transform(demographics: dict):
        # concatenates values by keys ordered alphabetically
        return torch.cat([torch.as_tensor(d[k], dtype=torch.float32).flatten() for k in sorted(d)])

        training_transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(common_shape), NormalizeIntensity(),
        ])

        target_transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(common_shape),
            AsDiscrete(to_onehot=2),
        ])

        dataset = MedicalFolderDataset(
            data_modalities=['T1', 'demographics'],
            target_modalities='label',
            transform={'T1': training_transform, 'demographics': demographics_transform},
            target_transform={'label': target_transform },
        )
        
        return DataManager(dataset, **loader_args)
    ...
```

For comprehensive transformation documentation, see [Applying Transformations](applying-transformations.md).

## Error Handling and Validation

The dataset provides comprehensive error handling:

- **Folder structure validation**: Ensures the required `subject/modality/file` structure
- **File format validation**: Checks for valid NIfTI files (.nii, .nii.gz)
- **Modality consistency**: Validates that all subjects have the required modalities
- **Demographics matching**: Ensures demographic data matches available subjects
- **Transform validation**: Validates transform compatibility with specified modalities

## Best Practices

Cross-Site Federated Deployment

- **Use Data Loading Plans**: Handle modality naming variations across different clinical sites
- **Standardize preprocessing**: Ensure consistent image preprocessing across all nodes
- **Consider scanner differences**: Account for variations in MRI scanners and acquisition protocols

Performance and Memory Management

- **Optimize batch sizes**: Medical images are large; use appropriate batch sizes (typically 1-4)
- **Efficient transforms**: Implement preprocessing efficiently to avoid memory bottlenecks
- **Consider resolution**: Balance image resolution with computational constraints

Clinical Considerations

- **Privacy compliance**: Ensure all medical data handling complies with regulations
- **De-identification**: Verify that all patient identifiers are properly removed
