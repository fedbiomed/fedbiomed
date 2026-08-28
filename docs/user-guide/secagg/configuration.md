# Secure Aggregation Configuration

Secure aggregation is implemented in Fed-BioMed and can be activated or deactivated as an option through the configuration. Even if secure aggregation is not configured during the initial installation, Fed-BioMed still works as long as the researcher or node components don't use it.

## Activating, Deactivating and Forcing Secure Aggregation

Nodes have the privilege of activating, deactivating, and enforcing secure aggregation. This means that model parameters can either be encrypted (required), optionally encrypted, or unencrypted. If a node requires model encryption and a training request from a researcher does not include secure aggregation context, the node will refuse training. If secure aggregation is allowed but not forced by the node, end-users are able to send training requests with or without secure aggregation.

In a federated setup, if one of the nodes requires secure aggregation but the researcher does not activate it, the FL round fails. Please refer to the researcher secure aggregation interface for more details.

!!! warning "Minimum number of nodes"
    Secure aggregation requires **at least 2 nodes** (plus the researcher). With a single node the aggregate would be that node's own value, and masking schemes have no peer to mask against, so the setup is rejected with a clear error. This applies to both federated training and federated analytics.

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

## Node-to-node channel keys

There is no default node-to-node private key to configure or replace. When a
secure aggregation exchange first needs a channel to a peer, each node creates
and stores its own P-256 key pair for that peer. The nodes exchange their public
channel keys using plaintext `ChannelSetupRequest` requests and
`ChannelSetupReply` responses relayed by the researcher. Subsequent LOM and
Joye-Libert setup messages are signed and encrypted using keys derived from the
per-peer ECDH agreement.

The public-key setup is automatic and requires no certificate registration.
It assumes an honest-but-curious researcher that forwards the public keys
without replacing them. See [Node-to-node overlay channel](./introduction.md#node-to-node-overlay-channel)
for the message flow and security boundary.


## Low-Overhead Masking (LOM)

LOM relies on a secure-aggregation-specific pairwise key setup using the Diffie-Hellman algorithm. These LOM keys are exchanged inside the encrypted node-to-node overlay and are distinct from the channel keys. It is possible to activate or force secure aggregation on the node side as explained in [Activating, Deactivating and Forcing Secure Aggregation](#activating-deactivating-and-forcing-secure-aggregation). Fed-BioMed sets up the pairwise LOM keys automatically under the honest-but-curious security model.

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

JLS is an HE-based secure aggregation scheme that relies on generating keys for encryption and decryption. Fed-BioMed automatically generates the keying material using the Additive Secret Sharing algorithm under the honest-but-curious researcher/server assumption. Individual node-to-node shares are sent through the encrypted overlay after its plaintext public-key setup. A passive researcher can relay but cannot decrypt these messages; an active researcher that substitutes channel setup keys is outside this assumption.
