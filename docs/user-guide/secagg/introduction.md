# Secure Aggregation

Fed-BioMed offers a secure aggregation framework where local model parameters of each node are encrypted before sending
them to the researcher/aggregator for aggregation. Parameters are encrypted using homomorphic encryption that
ties the decryption of the parameters to the execution of fixed computations. This guarantees that the model parameters will remain secure
on aggregator level, and researcher (component) or/and end-user will only have the final decrypted aggregated parameters.

## Available Secure Aggregation Schemes

Fed-BioMed supports two different secure aggregation schemes: Joye-Libert and Low-Overhead Masking (LOM). While Joye-Libert is an additive homomorphic encryption scheme, LOM is based on a masking approach.

### Low-Overhead Masking (LOM)

LOM is a secure aggregation scheme based on masking. It protects user inputs by applying masks using pairwise keys agreed upon by participating nodes. These pairwise keys are applied to private inputs in such a way that when all user inputs are summed at the aggregator level, the masks cancel out. One advantage of this scheme is that it does not require a setup phase for each training round, leading to faster training rounds.

#### Process flow

LOM consists of three phases: setup, protect, and aggregate. The setup phase is triggered by the first training request, the protect phase is applied on each node after training, and aggregation is performed on the researcher's side over protected/encrypted model weights or other inputs, such as auxiliary variables, depending on the type of training, optimizer, etc.

During the setup phase, all participating nodes agree on pairwise secrets using the Diffie-Hellman key exchange. These pairwise key agreements are established once per experiment, meaning that the agreed-upon keys can be used for multiple rounds within the same experiment. If a new node joins the experiment, all other participating nodes perform a key agreement with the new node. All these operations are coordinated by the researcher component.

Before the first training round, the researcher checks if the secure aggregation context (pairwise keys) is set up on the nodes. If not, the researcher sends secure aggregation setup requests. With this request, nodes receive a list of all nodes that will participate in the training, as well as the scheme that will be used for secure aggregation. Depending on the secure aggregation scheme (e.g., LOM), each node sends key agreement requests to the other participating nodes to create the pairwise secrets that will be used for the training.

Once all pairwise secrets are established, the researcher sends a train request to start the training. After each round of training, model weights are masked/encrypted using the pairwise keys (LOM scheme), so that summing all encrypted model weights of all the nodes will result in the unmasked aggregated model weights.


### Joye-Libert Secure Aggregation Scheme

Secure aggregation in Fed-BioMed is achieved through the use of a mix of the Joye-Libert (JL) aggregation scheme and the Shamir multi-party computation (MPC) protocol. JL is an additively homomorphic encryption (AHE) technique that encrypts model parameters locally using private and unique 2048-bit keys. The sum of encrypted model parameters can only be decrypted using the sum of private keys from each node that participate in the federated learning (FL) experiment. However, the encryption key used on each node is private and not shared with other parties or the central aggregator. Therefore, server key is calculated using MPC without revealing user (node) keys (server-key shares).

#### Shamir MPC Protocol

Shamir multi-party computation protocol is used to compute the server key, which is equal to the negative sum of
nodes' keys and is used for decryption on the researcher component. Thanks to MPC, the server key generation does not reveal nodes' private keys to the aggregator.


#### Process-flow

Since FL experiments are launched through researcher component, activating secure aggregation and setting up necessary
context is done through `Experiment` class of researcher component. However, the status of the secure aggregation
can be managed by node as well: node owner can disable, enable or force secure aggregation (see secure aggregation
node configuration for more details).

#### Technologies

##### MP-SPDZ

Fed-BioMed uses [MP-SPDZ](https://github.com/data61/MP-SPDZ) library for multi-party computation by launching MP-SPDZ
protocols at each secure aggregation context setup request. [MP-SPDZ](https://github.com/data61/MP-SPDZ) processes are
started at each request and stopped after context setup computation is completed.

##### Fault Tolerant Secure Aggregation

Fed-BioMed uses a modified version of  Joye-Libert scheme implementation from
repository [fault-tolerant-secure-agg](https://github.com/MohamadMansouri/fault-tolerant-secure-agg).



##### 1. Generating Public Parameter Biprime

At the beginning of the FL experiment, researcher sends secure aggregation context setup request to every node that
participates to this experiment. The first request is for generating public parameter `Biprime`. `Biprime` is
multiplication of two prime numbers that are generated using multi party computation. Final prime number is public
while prime shares are private and used for `Biprime` calculation. `Biprime` should be calculated using at least two
different parties. It is used for encrypting and decrypting aggregated parameters in Joye-Libert AHE.

!!! note "Current implementation"
    Since `Biprime` is public parameter, Fed-BioMed currently uses a default pre-generated 1024-bits biprime. Dynamic biprime
    generation on our road-map of future releases.

##### 2. Generating random key that are double the length of biprime

After biprime is generated or a default one is loaded, researcher sends another request for generating private key of
each node and the corresponding server key for researcher component. Each node generates random private keys.

!!! note "Key-size"
    Key size depends on biprime number that is used for secure aggregation. Maximum key-size should be less or equal
    the double of biprime key-size.

##### 3. Execute Shamir

Once the local key generation is completed, each node launches Shamir protocol to calculate negative sum of keys.
The result is only revealed it to the researcher.
This protocol is launched using [MP-SPDZ](#mp-spdz) library.

##### 4. Encrypting model parameters

If secure aggregation is activated for the `Experiment`, the training request contains information about which secure aggregation
context will be used for encryption. Once training is completed, model parameters are encrypted using biprime
and the user (node)-key.

##### 5. Decrypting sum of encrypted model parameters

After the encryption is done and the encrypted model parameters are received by the researcher, all model parameters
are aggregated using JL scheme. Aggregated parameters are clear text. Aggregation and decryption can not be performed
independently. It is an atomic operation.

!!! warning "Important"
    Secure aggregation requires at least 3 parties in FL experiment with one researcher and 2 nodes.


## Conclusions

Joye-Libert was the first secure aggregation algorithm implemented in Fed-BioMed. Later, the LOM scheme was introduced to simplify certain operations, such as the pre-setup phase and encryption processing time. While there are similarities between the two, there are also key differences.

In LOM, unlike Joye-Libert, secure aggregation does not require parties to perform certificate registration. Communication among the nodes is managed by the researcher using an honest-but-curious security model. This approach eliminates the need for the complicated and time-consuming pre-setup of Fed-BioMed nodes, where each party manually registers the certificates of other parties.

Another difference between the two schemes is that Joye-Libert requires the server/aggregator to possess a key to aggregate encrypted model inputs. In contrast, LOM does not require the aggregator to have an encryption key; the sum of the encrypted inputs directly results in the sum of the inputs. This makes Joye-Libert preferable in scenarios where it is necessary to explicitly identify a party that is allowed to perform the aggregation. In LOM, any party with access to all the masked inputs can obtain the aggregated inputs. This is not a concern in setups where all parties have equal rights to access the aggregated inputs, as long as communication between nodes and the researcher is secure which is the case by default in Fed-BioMed.

Neither algorithm is tolerant to dropouts. However, in LOM, if a new node joins the next round of training, all other nodes perform pairwise key setup with the new node. If one or more nodes drop out, there is no need to re-establish pairwise keys. In Joye-Libert, regardless of whether a new node joins or some nodes drop out, all keys must be regenerated.

In terms of encryption and aggregation processing time, LOM is significantly faster in cross-silo federated learning setups.

The security model implemented in Fed-BioMed's secure aggregation primarily targets the honest-but-curious parties' scenario, which applies to both algorithms.

## Next steps

- [Joye-Libert Scheme (JLS) Configuration](./configuration.md) documentation for more detail about configuring Fed-BioMed
  instances for secure aggregation.

- [Certificate and party registration for JLS](./certificate-registration.md) for MPC.

- [Activating secure aggregation](./researcher-interface.md) for the training through researcher component.

