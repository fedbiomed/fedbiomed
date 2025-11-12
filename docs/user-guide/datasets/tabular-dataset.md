# Tabular Datasets

## Introduction

Tabular datasets in Fed-BioMed handle structured data in tabular formats for classification and regression tasks.

**Key Features:**
- Automatic data loading with delimiter detection
- Format conversion (pandas, NumPy, PyTorch tensors)
- Framework compatibility (PyTorch and scikit-learn)

## Data Structure

**Basic CSV:**
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

Register using the Fed-BioMed node CLI, see [Deploying Datasets](../nodes/deploying-datasets.md) for details.:

```bash
fedbiomed node dataset add
# 1. Select "csv"
# 2. Path: /path/to/your/data.csv
# 3. Tags: #tabular
```

### Researcher-Side

Access tabular datasets through [experiment configuration](../researcher/experiment.md):

```python
from fedbiomed.researcher.experiment import Experiment

# Select nodes with tabular classification datasets
experiment = Experiment(
    tags=['#tabular'],
    model=my_model,
    training_plan_class=MyTrainingPlan,
    training_args=training_args
)
```

## Integration with Training Plans

### PyTorch Training Plan for Tabular Data

```python
...
```

### Scikit-learn Training Plan for Tabular Data

```python
...
```

## Best Practices and Common Issues

- Verify all numerical columns contain only numbers
- Check for mixed data types in the same column
- Handle string representations of numbers
- Choose appropriate imputation strategies
- Consider the impact of missing data
- Validate features don't introduce data leakage
- Check for infinite or NaN values
