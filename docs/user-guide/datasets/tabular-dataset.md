# Tabular Datasets

## Introduction

Tabular datasets in Fed-BioMed handle structured data in tabular formats for classification and regression tasks.

**Key Features:**

- Automatic data loading with delimiter detection
- Format conversion (pandas, NumPy, PyTorch tensors)
- Framework compatibility (PyTorch and scikit-learn)

## Data Structure

Each row is a sample; each column is a feature or target.

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.1,4.3,6.5,1
```

## Data Preparation

1. Clean data (handle missing values, outliers)
2. Format as CSV for maximum compatibility
3. Validate data types in columns

## Deployment

### Node-Side

Register the CSV file with the node CLI, see [Deploying Datasets](../nodes/deploying-datasets.md):

```bash
fedbiomed node dataset add
# 1. Select "csv"
# 2. Path: /path/to/your/data.csv
# 3. Unique tags and description (e.g. #tabular)
```

### Researcher-Side

Access tabular datasets through [experiment configuration](../researcher/experiment.md):

```python
from fedbiomed.researcher.federated_workflows import Experiment

experiment = Experiment(
    tags=['#tabular'],
    training_plan_class=MyTrainingPlan,
    model_args=model_args,
    training_args=training_args,
)
```

## Integration with Training Plans

The `training_data` method builds a `TabularDataset`, selecting the input and target columns, and
wraps it in a `DataManager`.

!!! warning "Numeric columns only"
    Columns selected as input or target must contain numeric values. Encode or drop non-numeric
    columns before deployment, or convert them with a [transform](#transformations).

### PyTorch Training Plan

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager


class MyTrainingPlan(TorchTrainingPlan):

    def init_model(self):
        return self.Net(self.model_args())

    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import TabularDataset", "import torch"]

    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.fc1 = nn.Linear(model_args['in_features'], 5)
            self.fc2 = nn.Linear(5, model_args['out_features'])

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    def training_step(self, data, target):
        output = self.model().forward(data).float()
        return torch.sqrt(torch.nn.MSELoss()(output, target))

    def training_data(self):
        dataset = TabularDataset(
            input_columns=['feature1', 'feature2', 'feature3'],
            target_columns=['target'],
            target_transform=lambda x: x.float(),
        )
        return DataManager(dataset=dataset)
```

### Scikit-learn Training Plan

```python
import numpy as np

from fedbiomed.common.training_plans import FedSGDRegressor
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset


class SGDRegressorTrainingPlan(FedSGDRegressor):

    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import TabularDataset", "import numpy as np"]

    def training_data(self):
        dataset = TabularDataset(
            input_columns=['feature1', 'feature2', 'feature3'],
            target_columns=['target'],
        )
        return DataManager(dataset=dataset)
```

Full runnable examples: the
[PyTorch Used Cars tutorial](../../tutorials/pytorch/03_PyTorch_Used_Cars_Dataset_Example.ipynb)
and the [scikit-learn SGD regressor tutorial](../../tutorials/scikit-learn/02_sklearn_sgd_regressor_tutorial.ipynb).

## Transformations

`transform` is applied to the input features and `target_transform` to the target. Each receives one
sample in the framework's native type (`torch.Tensor` or `numpy.ndarray`) and must return the same type.

```python
dataset = TabularDataset(
    input_columns=['feature1', 'feature2', 'feature3'],
    target_columns=['target'],
    transform=lambda x: (x - mean) / std,
    target_transform=lambda x: x.float(),
)
```

See [Applying Transformations](applying-transformations.md) for details.

## Troubleshooting

- **Non-numeric column error**: encode or drop string/categorical columns, or convert them with a `transform`.
- **Column not found**: column-name selection is case-sensitive; integer indexes must be in range.
- **Shape/dtype mismatch**: ensure `in_features`/`n_features` matches the number of `input_columns`, and cast the target dtype to match the loss.
