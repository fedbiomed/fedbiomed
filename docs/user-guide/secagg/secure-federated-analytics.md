# Secure Federated Analytics

This document describes how to use secure aggregation with federated analytics in Fed-BioMed.

## Overview

Secure federated analytics extends the existing [secure aggregation](./secagg/introduction.md) framework to protect individual node statistics during federated analytics queries. With secure aggregation enabled:

- **Node-level results are encrypted** before leaving the node
- **Only global aggregated results are decrypted** on the researcher side
- **Individual contributions remain private** - the researcher cannot see per-node statistics

## Supported Statistics

The following statistics can be protected with secure aggregation:

| Statistic | Protection | Notes |
|-----------|------------|-------|
| `mean` | ✅ Full | Scalar value encrypted |
| `variance` | ✅ Full | Scalar value encrypted |
| `count` | ✅ Full | Scalar value encrypted |
| `min` | ✅ Full | Scalar value encrypted |
| `max` | ✅ Full | Scalar value encrypted |
| `histogram` | ⚠️ Partial | Bin edges in clear, counts encrypted |
| `quantile` | ⚠️ Partial | Derived from histogram |
| `shape` | ❌ Not supported | Image metadata only |

## Usage

### Basic Usage

Enable secure aggregation when creating an experiment:

```python
from fedbiomed.researcher.federated_workflows import Experiment

# Enable with default LOM scheme
exp = Experiment(tags=['adni'], secagg=True)

# Compute mean - automatically encrypted and decrypted
result = exp.analytics.mean(dataset_args={'col_names': ['AGE']})

# Global result is decrypted
print(result.global_stat('mean'))
```

### Using Different Schemes

```python
from fedbiomed.researcher.secagg import SecureAggregation, SecureAggregationSchemes

# Use LOM scheme (default)
exp = Experiment(tags=['adni'], secagg=True)

# Or use Joye-Libert scheme (requires certificate configuration)
secagg = SecureAggregation(
    scheme=SecureAggregationSchemes.JOYE_LIBERT,
    active=True
)
exp = Experiment(tags=['adni'], secagg=secagg)
```

### Computing Multiple Statistics

```python
exp = Experiment(tags=['adni'], secagg=True)

result = exp.analytics.compute_analytics(
    stats=['mean', 'variance', 'count', 'min', 'max'],
    dataset_args={'col_names': ['AGE', 'BMI']}
)

# Access global aggregated statistics
global_stats = result.global_stats()
print(global_stats)
```

### Histogram with Secure Aggregation

```python
exp = Experiment(tags=['adni'], secagg=True)

result = exp.analytics.compute_analytics(
    stats=['histogram'],
    dataset_args={
        'col_names': ['AGE'],
        'histogram_args': {'bins': 10, 'range': (50, 100)}
    }
)

hist = result.global_stat('histogram')
print(f"Bin edges (in clear): {hist['bin_edges']}")
print(f"Counts (decrypted): {hist['counts']}")
```

### Conditional Enable/Disable

```python
# Create without secagg first
exp = Experiment(tags=['adni'])

# Enable later
exp.analytics.set_secagg(True)

# Or disable
exp.analytics.set_secagg(False)
```

## Without Secure Aggregation

When secagg is disabled (default), you can access individual node results:

```python
exp = Experiment(tags=['adni'], secagg=False)

result = exp.analytics.mean(dataset_args={'col_names': ['AGE']})

# Access per-node results
for node_id in result.node_ids:
    print(f"Node {node_id}: {result.node_stats(node_id)}")

# And global result
print(result.global_stat('mean'))
```

## Security Properties

### What is Protected

- Individual node statistics are encrypted using the selected scheme (LOM or Joye-Libert)
- Encryption keys are shared among all participants
- Only the sum of encrypted values can be decrypted (not individual contributions)

### What is NOT Protected

- **Histogram bin edges**: Must be shared in clear for interpretation
- **Metadata**: Node selection, timing information
- **Dataset schema**: Column names and types

## Configuration

### Node Configuration

Ensure nodes have federated analytics enabled:

```python
# In node config (config.py)
allow_federated_analytics = True
```

### Security Validation

By default, secure aggregation includes validation to ensure correct decryption. This can be disabled:

```python
exp = Experiment(tags=['adni'], secagg=True)

# Validation is automatic - no extra configuration needed
```

## Comparison: With vs Without SecAgg

| Feature | With SecAgg | Without SecAgg |
|---------|-------------|----------------|
| Per-node results visible | ❌ No | ✅ Yes |
| Global result visible | ✅ Yes | ✅ Yes |
| Node privacy | ✅ Protected | ❌ Not protected |
| Setup complexity | Higher | Lower |
| Computational overhead | Higher | Lower |

## Testing

Run unit tests for secure federated analytics:

```bash
pytest tests/test_analytics/test_federated_analytics.py::TestSecureFederatedAnalytics -v
pytest tests/test_analytics/test_federated_analytics.py::TestFAJobEncryption -v
```

## See Also

- [Secure Aggregation User Guide](./secagg/introduction.md)
- [Federated Analytics Tutorial](../tutorials/federated-analytics.ipynb)
- [LOM Scheme Details](./secagg/lom.md)
- [Joye-Libert Scheme Details](./secagg/joye-libert.md)
