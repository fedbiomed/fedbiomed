# Secure Aggregation

Fed-BioMed offers a secure aggregation framework where local model parameters of each node are encrypted before sending
them to the researcher/aggregator for aggregation. Parameters are encrypted using homomorphic encryption that
ties the decryption of the parameters to the execution of fixed computations. This guarantees that the model parameters will remain secure
on aggregator level, and researcher (component) or/and end-user will only have the final decrypted aggregated parameters.

## Technologies

### MP-SPDZ

Fed-BioMed uses [MP-SPDZ](https://github.com/data61/MP-SPDZ) library for multi-party computation by launching MP-SPDZ
protocols at each secure aggregation context setup request. [MP-SPDZ](https://github.com/data61/MP-SPDZ) processes are
started at each request and stopped after context setup computation is completed.

### Fault Tolerant Secure Aggregation

Fed-BioMed uses a modified version of  Joye-Libert scheme implementation from
repository [fault-tolerant-secure-agg](https://github.com/MohamadMansouri/fault-tolerant-secure-agg).


## Methods and Techniques

### Joye-Libert Secure Aggregation Scheme

Secure aggregation in Fed-BioMed is achieved through the use of a mix of the Joye-Libert (JL) aggregation scheme and
the Shamir multi-party computation (MPC) protocol. JL is an additively homomorphic encryption (AHE) technique that encrypts model
parameters locally using private and unique 2048-bit keys. The sum of encrypted model parameters can only be
decrypted using the sum of private keys from each node that participate in the federated learning (FL) experiment.
However, the encryption key used on each node is private and not shared with other parties or the central aggregator.
Therefore, server key is calculated using MPC without revealing user (node) keys (server-key shares).

### Shamir MPC Protocol

Shamir multi-party computation protocol is used to compute the server key, which is equal to the negative sum of
nodes' keys and is used for decryption on the researcher component. Thanks to MPC, the server key generation does not reveal nodes' private keys to the aggregator.


### Process-flow

Since FL experiments are launched through researcher component, activating secure aggregation and setting up necessary
context is done through `Experiment` class of researcher component. However, the status of the secure aggregation
can be managed by node as well: node owner can disable, enable or force secure aggregation (see secure aggregation
node configuration for more details).


#### 1. Generating Public Parameter Biprime

At the beginning of the FL experiment, researcher sends secure aggregation context setup request to every node that
participates to this experiment. The first request is for generating public parameter `Biprime`. `Biprime` is
multiplication of two prime numbers that are generated using multi party computation. Final prime number is public
while prime shares are private and used for `Biprime` calculation. `Biprime` should be calculated using at least two
different parties. It is used for encrypting and decrypting aggregated parameters in Joye-Libert AHE.

!!! note "Current implementation"
    Since `Biprime` is public parameter, Fed-BioMed currently uses a default pre-generated 1024-bits biprime. Dynamic biprime
    generation on our road-map of future releases.

#### 2. Generating random key that are double the length of biprime

After biprime is generated or a default one is loaded, researcher sends another request for generating private key of
each node and the corresponding server key for researcher component. Each node generates random private keys.

!!! note "Key-size"
    Key size depends on biprime number that is used for secure aggregation. Maximum key-size should be less or equal
    the double of biprime key-size.

#### 3. Execute Shamir

Once the local key generation is completed, each node launches Shamir protocol to calculate negative sum of keys.
The result is only revealed it to the researcher.
This protocol is launched using [MP-SPDZ](#mp-spdz) library.

#### 4. Encrypting model parameters

If secure aggregation is activated for the `Experiment`, the training request contains information about which secure aggregation
context will be used for encryption. Once training is completed, model parameters are encrypted using biprime
and the user (node)-key.

#### 5. Decrypting sum of encrypted model parameters

After the encryption is done and the encrypted model parameters are received by the researcher, all model parameters
are aggregated using JL scheme. Aggregated parameters are clear text. Aggregation and decryption can not be performed
independently. It is an atomic operation.

!!! warning "Important"
    Secure aggregation requires at least 3 parties in FL experiment with one researcher and 2 nodes.


## Conclusions

In Fed-BioMed, Joye-Libert secure aggregation has been chosen because it provides fast encryption and it does not require the re-generation
of server-key and user-key shares for every training round, making it faster. Although it is not tolerant to drop-out
during one round of training, the implementation allows the re-configuration of Joye-Libert elements for the
next round if a node is dropped or a new node is added.

The security model implemented in Fed-BioMed's secure aggregation primarily targets the case of honest but curious parties.


## Next steps

- [Configuration](./configuration.md) documentation for more detail about configuring Fed-BioMed
  instances for secure aggregation.

- [Certificate and party registration](./certificate-registration.md) for MPC.

- [Activating secure aggregation](./researcher-interface.md) for the training through researcher component.

