## Fed-BioMed v4.3 new release

![v4.3](../assets/img/v4.3.jpg#img-centered-sm)

Fed-BioMed v4.3 is now available.

It introduces **Secure Aggregation** functionality which further protects the federated learning process. Secure Aggregation encrypts model parameters sent by the nodes to the researcher. The researcher then computes aggregated model parameters but cannot access individual node's model parameters in cleartext.

Fed-BioMed Secure Aggregation uses Joye-Libert additively homomorphic encryption scheme (based on [fault-tolerant-secure-agg](https://github.com/MohamadMansouri/fault-tolerant-secure-agg) implementation) and Shamir multi-party computation protocol (from [MP-SPDZ](https://github.com/data61/MP-SPDZ) software).

Bug fixes and misc updates are also included in the release.

More details about the new features can be found in the [Fed-BioMed CHANGELOG](https://github.com/fedbiomed/fedbiomed/blob/v4.3/CHANGELOG.md).
