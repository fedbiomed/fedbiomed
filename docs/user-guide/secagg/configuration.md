# Secure Aggregation Configuration

Secure aggregation is implemented in Fed-BioMed and can be activated or deactivated as an option through the configuration. Even if secure aggregation is not configured during the initial installation, Fed-BioMed still works as long as the researcher or node components don't use it.

## Activating, Deactivating and Forcing Secure Aggregation

Nodes have the privilege of activating, deactivating, and enforcing secure aggregation. This means that model parameters can either be encrypted (required), optionally encrypted, or unencrypted. If a node requires model encryption and a training request from a researcher does not include secure aggregation context, the node will refuse training. If secure aggregation is allowed but not forced by the node, end-users are able to send training requests with or without secure aggregation.

In a federated setup, if one of the nodes requires secure aggregation but the researcher does not activate it, the FL round fails. Please refer to the researcher secure aggregation interface for more details.

!!! note "Researcher"
    Researcher configuration file does not have parameter regarding secure aggregation activation. However, secure aggregation context is managed through [Experiment][fedbiomed.researcher.federated_workflows.Experiment] interface (class).


Example: security section of the configuration file with secure aggregation optional.

```ini
[security]
hashing_algorithm = SHA256
allow_default_training_plans = True
training_plan_approval = False
secure_aggregation = True
force_secure_aggregation = False
```


## Low-Overhead Masking (LOM)

LOM scheme is secure aggregation relies on pair-wise key setup using Deffie-Hellman algorithm. It is possible to directly activate or force secure aggregation on the node side as it is explained in the section  [Activating, Deactivating and Forcing Secure Aggregation](#activating,-aeactivating-and-forcing-secure-aggregation). Pair-wise keys will be setup by Fed-BioMed automatically honest-but-curious security model.

Example: security section of the configuration file with secure aggregation mandatory.

```ini
[security]
hashing_algorithm = SHA256
allow_default_training_plans = True
training_plan_approval = False
secure_aggregation = True
force_secure_aggregation = True
```


## Joye-Libert Scheme (JLS)

JLS is an HE-based secure aggregation scheme that relies on generating keys for encryption and decryption. The keying material is automatically generated by Fed-BioMed in a secure manner using the Additive Secret Sharing algorithm, under the honest-but-curious researcher/server assumption. This secret sharing is conducted through encrypted node-to-node communication, ensuring that the server cannot decrypt the transmitted information.
