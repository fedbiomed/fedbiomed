# Fed-BioMed deployment scenarios

Fed-BioMed can be deployed in different ways:

* **single-machine** (nodes, network and researcher run on the same machine) or **multiple-machine**
* **with or without containers** (each component runs in its `docker` container) **and VPN** (all communications between components are tunneled in a WireGuard VPN with mutual authentication of the VPN endpoints)

Choose a scenario depending on the context and requirements:

* **Single-machine without VPN/containers is the [basic simple installation scenario](../../tutorials/installation/0-basic-software-installation.md)** described in the introduction tutorials. Use cases include: newcomer testing Fed-BioMed software ; FL researcher designing and testing FL methods with (non sensitive) data on the laptop ; software developer contributing to Fed-BioMed.

* Single-machine with VPN/containers deployment is [briefly described here](https://gitlab.inria.fr/fedbiomed/fedbiomed/-/tree/master/README.md#install-and-run-in-vpndevelopment-environment). The use case is the simplified testing of the VPN/containers facility (eg for testing purpose or integration tests).

* Multiple-machine without VPN/containers deployment is [briefly described here](https://gitlab.inria.fr/fedbiomed/fedbiomed/-/blob/master/README.md#run-the-node-part). This should only be used when components are connected through a highly secure network.

* **Multiple-machine with VPN/containers deployment [is documented here](./deployment-vpn.md)**. Most real-life deployments use this scenario to protect node data and communications between components. Typical deployment includes a secure federation server and one node for each data provider site.

Check the [security model](./security-model.md) and [network communications](./matrix.md) to understand which scenario fits your needs.
