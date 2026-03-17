# Federated Analytics — Nodes

## Overview

This page is for **node operators**. It explains how to enable or disable Federated Analytics (FA) on a node and what that means in practice.

Federated Analytics lets researchers compute statistics across data held by multiple nodes **without the raw data ever leaving any node**. When a researcher sends an analytics request, each node computes the statistics locally and sends back only the aggregated summaries.

> - For which datasets support FA and how to make a custom dataset FA-compatible, see [Federated Analytics — Datasets](../datasets/federated-analytics.md).
> - For how to run analytics as a researcher, see [Federated Analytics — Researcher](../researcher/federated-analytics.md).

---

## Enabling or Disabling FA

Federated Analytics is **enabled by default** on every node. The setting lives in the `[security]` section of the node configuration file:

```ini
[security]
allow_federated_analytics = True
```

To disable FA on a node, set the value to `False`:

```ini
[security]
allow_federated_analytics = False
```

Once a dataset is registered on a node with the appropriate tags, it is automatically eligible for analytics requests — no further configuration is needed beyond this flag.

!!! note "See also"
    [Configuring Nodes](configuring-nodes.md) for a full reference of all node configuration options.

---

## Common Errors & Troubleshooting

| Error message | Cause | Fix |
|---|---|---|
| `Federated Analytics are not allowed on this node` | The node has `allow_federated_analytics = False` in its config | Set `allow_federated_analytics = True` in the `[security]` section of the node configuration file |
