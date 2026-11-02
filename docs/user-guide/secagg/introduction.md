# Secure Aggregation

Fed-BioMed offers a secure aggregation framework where local model parameters of each node are encrypted before sending
them to the researcher/aggregator for aggregation. Decryption of the parameters is tied to the execution of fixed computations. This guarantees that the local model parameters will remain private
on aggregator level, and researcher (component) or/and end-user will only have the final decrypted aggregated parameters.

## Available Secure Aggregation Schemes

Fed-BioMed supports two different secure aggregation schemes: Joye-Libert and Low-Overhead Masking (LOM). While Joye-Libert is an additive homomorphic encryption scheme, LOM is based on a masking approach.

## Node-to-node overlay channel

Both secure aggregation schemes exchange setup material through Fed-BioMed's
logical node-to-node overlay. Nodes do not open a direct network connection to
one another: the researcher relays the overlay messages over the existing gRPC
connections.

The first message to a peer sets up the overlay channel as follows:

1. Each node creates a distinct NIST P-256 key pair for that peer. The local
   private key is stored in the node's channel database.
2. A plaintext `ChannelSetupRequest` asks for the peer's public channel key,
   and a plaintext `ChannelSetupReply` returns it. These messages contain no
   encryption salt or nonce. The public key is not secret, but this initial
   exchange is not authenticated end-to-end by the overlay protocol.
3. For each later overlay message, the sender performs ECDH with its local
   channel private key and the peer's public channel key. A fresh salt and the
   ordered node identifiers are used to derive a 32-byte message key. The inner
   message is signed with ECDSA and encrypted with ChaCha20. The salt and nonce
   travel in the outer overlay message; they are public parameters and do not
   reveal the derived key.

!!! warning "Channel setup trust assumption"
    The researcher can observe the plaintext public channel keys, salts,
    nonces, and encrypted overlay messages, but an honest-but-curious
    researcher cannot derive the ECDH message keys. Because the initial public
    keys are not authenticated end-to-end, an active relay that replaces those
    keys can perform a man-in-the-middle attack. The current secure aggregation
    channel therefore assumes that the researcher relays channel setup
    messages without modification.

The overlay channel keys only protect node-to-node messages in transit. They
are separate from the LOM pairwise masking keys and the Joye-Libert private
keys and secret shares described below.

### Low-Overhead Masking (LOM)

LOM is a secure aggregation scheme based on masking. It protects user inputs by applying masks using pairwise keys agreed upon by participating nodes. These pairwise keys are applied to private inputs in such a way that when all user inputs are summed at the aggregator level, the masks cancel out. One advantage of this scheme is that it does not require a setup phase for each training round, leading to faster training rounds.

#### Process flow

LOM consists of three phases: setup, protect, and aggregate. The setup phase is triggered by the first training request, the protect phase is applied on each node after each training, and aggregation is performed on the researcher's side over protected/encrypted model weights or other inputs, such as auxiliary variables, depending on the type of training, optimizer, etc.

During the setup phase, all participating nodes agree on pairwise secrets using a second, secure-aggregation-specific Diffie-Hellman key exchange. Its `KeyRequest` and `KeyReply` messages are carried inside the encrypted node-to-node overlay channel described above. The resulting LOM pairwise secrets are distinct from the overlay channel keys and are established once per experiment, meaning that they can be used for multiple rounds within the same experiment. If a new node joins the experiment, all other participating nodes perform a key agreement with the new node. All these operations are coordinated by the researcher component.

Before the first training round, the researcher checks if the secure aggregation context (pairwise keys) is set up on the nodes. If not, the researcher sends secure aggregation setup request to the nodes. With this request, nodes receive a list of all nodes that will participate in the training, as well as the scheme that will be used for secure aggregation. Depending on the secure aggregation scheme (e.g., LOM), each node sends key agreement requests to the other participating nodes to create the pairwise secrets that will be used for the training.

Once all pairwise secrets are established, the researcher sends a train request to start the training. After each round of training, model weights are masked/encrypted using the pairwise keys (LOM scheme), so that summing all encrypted model weights of all the nodes will result in the unmasked aggregated model weights.


### Joye-Libert Secure Aggregation Scheme

Secure aggregation in Fed-BioMed is achieved through the use of a mix of the Joye-Libert (JL) aggregation scheme and the Shamir multi-party computation (MPC) protocol. JL is an additively homomorphic encryption (AHE) technique that encrypts model parameters locally using private and unique 2048-bit keys. The sum of encrypted model parameters can only be decrypted using the sum of private keys from each node that participate in the federated learning (FL) experiment. However, the encryption key used on each node is private and not shared with other parties or the central aggregator. Therefore, server key is calculated using MPC without revealing user (node) keys (server-key shares).

#### Additive Secret Sharing Protocol

Additive secret sharing is a protocol used to compute the server key, which is equal to the negative sum of the nodes' keys and is used for decryption on the researcher component. This secret-sharing-based MPC algorithm computes the sum of the nodes' keys without revealing the nodes' private keys to the aggregator.

#### Technologies

##### Fault Tolerant Secure Aggregation

Fed-BioMed uses a modified version of  Joye-Libert scheme implementation from repository [fault-tolerant-secure-agg](https://github.com/MohamadMansouri/fault-tolerant-secure-agg).


#### Process-flow

Since FL experiments are launched through researcher component, activating secure aggregation and setting up necessary context is done through `Experiment` class of researcher component. However, the status of the secure aggregation can be managed by node as well: node owner can disable, enable or force secure aggregation (see secure aggregation node configuration for more details).


##### 1. Public Parameter Biprime

`Biprime` is multiplication of two prime numbers. Prime number is public while prime shares are private and used for `Biprime` calculation. Fed-BioMed uses securely generated default static biprime which is located in `envs/common/default_primes/biprime0.json`.


##### 2. Generating random key that are double the length of biprime

Researcher sends a request for generating private key of each node and the corresponding server key for researcher component. Each node generates random private keys.

!!! note "Key-size"
    Key size depends on biprime number that is used for secure aggregation. Maximum key-size should be less or equal the double of biprime key-size.

##### 3. Execute Additive Secret Sharing


The nodes that receive a secure aggregation setup request launch the Additive Secret Sharing algorithm to calculate the server key used for aggregating protected private data. Each node's individual share for another node is sent in an `AdditiveSSharingReply` through the established encrypted node-to-node overlay channel. The Joye-Libert key and its shares are not used as transport-encryption keys.


##### 4. Encrypting model parameters

If secure aggregation is activated for the `Experiment`, the training request contains information about which secure aggregation context will be used for encryption. Once training is completed, model parameters are encrypted using biprime and the user (node)-key.

##### 5. Decrypting sum of encrypted model parameters

After the encryption is done and the encrypted model parameters are received by the researcher, all model parameters are aggregated using JL scheme. Aggregated parameters are clear text. Aggregation and decryption can not be performed independently. It is an atomic operation.

!!! warning "Important"
    Secure aggregation requires at least 3 parties in FL experiment with one researcher and 2 nodes.


## Conclusions

Joye-Libert was the first secure aggregation algorithm implemented in Fed-BioMed. Later, the LOM scheme was introduced to simplify certain operations, such as the pre-setup phase and speed up encryption processing. While there are similarities between the two, there are also key differences.

Secure aggregation does not require parties to provision a shared default node-to-node key. Each node automatically creates per-peer overlay channel keys, and public keys are exchanged when the channel is first used. Communication among the nodes is relayed by the researcher using the honest-but-curious assumption described above. This avoids distributing shared private key material or requiring manual registration of peer certificates for the overlay channel.

Another difference between the two schemes is that Joye-Libert requires the server/aggregator to possess a key to aggregate encrypted model inputs. In contrast, LOM does not require the aggregator to have an encryption key; the sum of the encrypted inputs directly results in the sum of the inputs. This makes Joye-Libert preferable in scenarios where it is necessary to explicitly identify a party that is allowed to perform the aggregation. In LOM, any party with access to all the masked inputs can obtain the aggregated inputs. This is not a concern in setups where all parties have equal rights to access the aggregated inputs.

Neither algorithm is tolerant to dropouts. However, in LOM, if a new node joins the next round of training, all other nodes perform pairwise key setup with the new node. If one or more nodes drop out, there is no need to re-establish pairwise keys. In Joye-Libert, regardless of whether a new node joins or some nodes drop out, all keys must be regenerated.

In terms of encryption and aggregation processing time, LOM is significantly faster in most cross-silo federated learning setups, typically with increasing model parameters size and number of nodes no more than a few dozens.

The security model implemented in Fed-BioMed's secure aggregation primarily targets the honest-but-curious parties' scenario, which applies to both algorithms.

## Next steps

- [Joye-Libert Scheme (JLS) Configuration](./configuration.md) documentation for more detail about configuring Fed-BioMed
  instances for secure aggregation.

- [Activating secure aggregation](./researcher-interface.md) for the training through researcher component.
