# Network communication matrix

This page describes the network communications:

- between the Fed-BioMed components (`node`s, and `researcher`), aka application internal / backend communications
- for user access to the Fed-BioMed components GUI
- for Fed-BioMed software installation and docker image build

Communications between the components depend on the [deployment scenario](./deployment.md): with or without VPN/containers.

Network communications for the setup of host machines and the installation of [software requirements](../../getting-started/installation.md) are out of the scope of this document. Network communications necessary for the host machines to run (access to DNS, LDAP, etc.) regardless of Fed-BioMed are also out of the scope of this document. They are specific to system configuration and installation processes.


## Introduction

Fed-BioMed network communications basic principle is that all communications between components are outbound from one `node` to the `researcher`. There is no inbound communications to a `node`.

The exception to this principle are optional direct communications between the components for using the secure aggregation feature (eg. `node`/`researcher` to `node`/`researcher` communication for cryptographic material negotiation). The communications for crypto material are closed after the negotiation is completed and handle only secagg key negotiation requests.

Fed-BioMed provides some optional GUI for the `node` (node configuration GUI) and the `researcher` (Jupyter notebook and Tensorboard).
By default, these GUI components are not secured (no HTTPS and/or no certificate signed by well known authority). So they are configured by default to accept only communications from the same machine (*localhost*).


## Software installation

Network communications for software installation cover the Fed-BioMed [software installation](../../getting-started/installation.md) and setup until the software is ready to be used.

They cover all [deployment scenarios](./deployment.md). If multiple machines are used, each machine needs to authorize these communications.

* direction is *out* (outbound communication from the component) or *in* (inbound communication to the component)

| dir | source machine | destination machine | destination port | service |
| --  | ------------   | -----------------   | --------------   | -----   |
| out | component      | Internet            | TCP/80           | HTTP    |
| out | component      | Internet            | TCP/443          | HTTPS   |

<br>

"Component" is the `node` or `researcher`.

For destination machine, it is simpler to authorize outbound communications to all Internet addresses for the required ports during installation. Indeed, several packaging systems are used for installation, with no guarantee of stable IP address used by the packaging server:

- For all [deployment scenarios](./deployment.md): conda, pip (all components) and yarn/npm (node GUI component) packages
- Plus for VPN/containers scenarios: dockerhub images, apt apk and cargo packages, git over https cloning, wget and curl download

Note: when using a VPN/containers scenario, a site with very stringent requirements on `node`'s communication can avoid authorizing the above communications for installation of the node components (`node` and `gui`). For that, it needs to build the components docker image on another machine (with the above filter), save the image, copy it to the node machine, load it on the node machine. This scenario is not fully packaged and documented by Fed-BioMed but you can find [some guidelines here](https://github.com/fedbiomed/fedbiomed/blob/master/envs/vpn/README.md#specific-instructions-building-node-image-on-a-different-machine).


## Running without VPN/containers

![fedbiomed-network-matrix](../../assets/img/fedbiomed_matrix_native.png#img-centered-lr)


This part describes the communication matrix for running Fed-BioMed without VPN/containers:

* The direction is *out* (outbound communication from the component) or *in* (inbound communication to the component)
* Type of communication is either *backend* (between the application components) or *user* (user access to a component GUI). Command line user access to component from *localhost* are not noted here. GUI access are noted though recommended default configuration is to give access only from *localhost*
* The status is either *mandatory* (needed to run Fed-BioMed) or *optional* (a Fed-BioMed experiment can run without this part)

On the node component (`node` + `gui` ):

| dir | source machine | destination machine | destination port | service   | type     | status    | comment  |
| --  | ------------   | -----------------   | --------------   | -----     | ------   | --------  | ------   |
| out | node           | researcher             | TCP/50051     | gRPC/TLS  | backend  | mandatory | node-researcher communications |
| in  | *localhost*    | gui                 | TCP/8484         | HTTP      | user     | optional  | node GUI |

* `node` and `gui` also need a shared filesystem, so they are usually installed on the same machine.

<br>

On the researcher component (`researcher`):

| dir | source machine | destination machine | destination port | service   | type     | status    | comment     |
| --  | ------------   | -----------------   | --------------   | -----     | ------   | --------  | -----       |
| in  | *nodes*        | researcher          | TCP/50051        | gRPC/TLS  | backend  | mandatory | node-researcher communications|
| in  | *localhost*    | researcher          | TCP/8888         | HTTP      | user     | optional  | Jupyter     |
| in  | *localhost*    | researcher          | TCP/6006         | HTTP      | user     | optional  | Tensorboard |


## Running with VPN/containers

![fedbiomed-network-matrix](../../assets/img/fedbiomed_matrix_vpn.png#img-centered-lr)

This part describes the communication matrix for running Fed-BioMed with VPN/containers:

* The direction is *out* (outbound communication from the component) or *in* (inbound communication to the component)
* Type of communication is either *backend* (between the application components) or *user* (user access to a component GUI). Command line user access to component from *localhost* are not noted here. GUI access are noted though recommended default configuration is to give access only from *localhost*
* The status is either *mandatory* (needed to run Fed-BioMed) or *optional* (a Fed-BioMed experiment can run without this part)

On the node component (`node` + `gui` ):

| dir | source machine | destination machine | destination port | service   | type     | status    | comment  |
| --  | ------------   | -----------------   | --------------   | -----     | ------   | --------  | ------   |
| out | node           | vpnserver           | UDP/51820        | WireGuard | backend  | mandatory |          |
| in  | *localhost*    | gui                 | TCP/8443         | HTTPS     | user     | optional  | node GUI |

* `node` and `gui` also need a shared filesystem, so they are usually installed on the same machine.

<br>

On the researcher component (`researcher`):

| dir | source machine | destination machine | destination port | service   | type     | status    | comment     |
| --  | ------------   | -----------------   | --------------   | -----     | ------   | --------  | -----       |
| out | researcher     | vpnserver           | UDP/51820        | WireGuard | backend  | mandatory |             |
| in  | *localhost*    | researcher          | TCP/8888         | HTTP      | user     | optional  | Jupyter     |
| in  | *localhost*    | researcher          | TCP/6006         | HTTP      | user     | optional  | Tensorboard |

