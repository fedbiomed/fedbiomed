#!/usr/bin/env python3
"""
Example: Secure Federated Analytics

This script demonstrates how to use secure aggregation with federated analytics
in Fed-BioMed. Secure aggregation ensures that individual node statistics are
encrypted and only the global aggregated result is visible to the researcher.

Prerequisites:
1. At least 2 nodes running with datasets
2. Nodes should have `allow_federated_analytics` enabled (default)
3. For Joye-Libert scheme, proper certificate configuration is needed

Usage:
    # Basic usage with default LOM scheme
    python secure_federated_analytics_example.py

    # With explicit Joye-Libert scheme
    # (requires proper certificate configuration)
"""

from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.secagg import SecureAggregation, SecureAggregationSchemes


def example_basic_secure_analytics():
    """Example 1: Basic secure federated analytics with default LOM scheme."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Secure Federated Analytics")
    print("=" * 60)

    # Create experiment with secure aggregation enabled
    # Using secagg=True uses the default LOM (Learning with Optical Multiphoton) scheme
    tags = ['adni']  # Use your dataset tags
    exp = Experiment(
        tags=tags,
        secagg=True  # Enable secure aggregation
    )

    print(f"\nExperiment created with SecAgg: {exp.analytics.secagg}")
    print(f"SecAgg active: {exp.analytics.secagg.active}")

    # Fetch a FAResult object to access global_stat and available_stats.
    # This will:
    # 1. Setup secure aggregation context with nodes
    # 2. Send encrypted requests to nodes
    # 3. Receive encrypted responses
    # 4. Decrypt and aggregate on researcher side
    result = exp.analytics.fetch_stats(stats='mean', dataset_schema=['AGE'])

    # The global result is automatically decrypted
    print(f"\nGlobal mean AGE: {result.global_stat('mean')}")
    print(f"Available stats: {result.available_stats()}")


def example_explicit_scheme():
    """Example 2: Secure federated analytics with explicit scheme selection."""
    print("\n" + "=" * 60)
    print("Example 2: Explicit Scheme Selection")
    print("=" * 60)

    # Create SecureAggregation with explicit scheme
    secagg = SecureAggregation(
        scheme=SecureAggregationSchemes.LOM,  # or Joye-Libert
        active=True
    )

    exp = Experiment(
        tags=['adni'],
        secagg=secagg  # Pass pre-configured SecureAggregation
    )

    result = exp.analytics.fetch_stats(stats='mean', dataset_schema=['AGE'])
    print(f"Global mean: {result.global_stat('mean')}")


def example_multiple_stats():
    """Example 3: Computing multiple statistics securely."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Statistics")
    print("=" * 60)

    exp = Experiment(
        tags=['adni'],
        secagg=True
    )

    # Compute multiple statistics at once
    stats = ['mean', 'variance', 'count', 'min', 'max']
    result = exp.analytics.fetch_stats(
        stats=stats,
        dataset_schema=['AGE']
    )

    print("\nGlobal statistics for AGE:")
    global_stats = result.global_stats()
    for col, col_stats in global_stats.items():
        print(f"\n{col}:")
        for stat_name, value in col_stats.items():
            print(f"  {stat_name}: {value}")


def example_histogram_secure():
    """Example 4: Secure histogram computation."""
    print("\n" + "=" * 60)
    print("Example 4: Secure Histogram")
    print("=" * 60)

    exp = Experiment(
        tags=['adni'],
        secagg=True
    )

    # Histogram bins are sent in clear (for interpretation)
    # but counts are encrypted
    result = exp.analytics.fetch_stats(
        stats=['histogram'],
        dataset_schema=['AGE'],
        stats_args={
            'histogram_args': {
                'bins': 10,
                'range': (50, 100)
            }
        }
    )

    hist = result.global_stat('histogram')
    # global_stat preserves the column-nested structure: {'AGE': {'bin_edges': ..., 'counts': ...}}
    print(f"\nHistogram bin_edges: {hist.get('AGE', {}).get('bin_edges', [])}")
    print(f"Histogram counts: {hist.get('AGE', {}).get('counts', [])}")


def example_disable_secagg():
    """Example 5: Disable secure aggregation (for comparison)."""
    print("\n" + "=" * 60)
    print("Example 5: Compare with Non-Secure Mode")
    print("=" * 60)

    # Without secagg - raw values are visible
    exp = Experiment(
        tags=['adni'],
        secagg=False  # Disable secure aggregation
    )

    result = exp.analytics.fetch_stats(stats='mean', dataset_schema=['AGE'])

    # Can access individual node results
    print("\nPer-node statistics (visible without SecAgg):")
    for node_id in result.node_ids:
        print(f"  {node_id}: {result.node_stats(node_id)}")

    print(f"\nGlobal mean: {result.global_stat('mean')}")


def example_conditional_secagg():
    """Example 6: Conditional secure aggregation."""
    print("\n" + "=" * 60)
    print("Example 6: Conditional SecAgg")
    print("=" * 60)

    # Create experiment without secagg initially
    exp = Experiment(tags=['adni'])

    # Enable secagg later if needed
    exp.analytics.set_secagg(True)

    print(f"SecAgg enabled: {exp.analytics.secagg.active}")

    result = exp.analytics.fetch_stats(stats='mean', dataset_schema=['AGE'])
    print(f"Global mean: {result.global_stat('mean')}")


if __name__ == "__main__":
    print("Secure Federated Analytics Examples")
    print("=" * 60)
    print("""
These examples demonstrate secure aggregation for federated analytics.

Note: These examples require:
1. Running nodes with datasets
2. Network connectivity between researcher and nodes
3. For Joye-Libert: proper certificate configuration

To run these examples:
1. Start your nodes: fedbiomed node -p my-node start
2. Start the researcher component
3. Modify the tags to match your dataset
4. Uncomment the example functions to run

For testing without live nodes, use the unit tests:
    pytest tests/test_analytics/test_federated_analytics.py::TestSecureFederatedAnalytics -v
""")

    # Uncomment to run examples:
    # example_basic_secure_analytics()
    # example_explicit_scheme()
    # example_multiple_stats()
    # example_histogram_secure()
    # example_disable_secagg()
    # example_conditional_secagg()
