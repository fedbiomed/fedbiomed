---
title: Node Configuration
description: Configuration instructions for Fed-BioMed nodes. 
keywords: fedbiomed configuration,node configuration
---

# Node Configuration

Fed-BioMed framework has 2 main components: `node` and `researcher`. A `node` stores private datasets and 
performs training in response to researcher's train requests. It communicates directly with the `researcher` component using RPC protocol.


A basic node configuration contains the following settings:

- providing a python environment
- assigning a unique node id 
- security parameters such as training plan approval mode, hashing algorithms 
- connection credentials for the researcher component

!!! note "Note"
    These basic configurations are created automatically using default values by the scripts provided in Fed-BioMed.
    While it is possible to modify the configuration manually through configuration file, some parameters may become incompatible upon doing so; such as [server]/host and [server]/pem for the latter depends on the former.
    It is therefore strongly adviced to rely on the dedicated script for configuration creation, namely : `fedbiomed_run configuration create`; whose options are described above.
    
## Environment for Nodes 

A `node` requires a conda environment to be able to run. This environment provides necessary python modules for both the task management part and the model training part. Thanks to Fed-BioMed Node CLI, this conda environment can be created easily. 

```
$ ${FEDBIOMED_DIR}/scripts/configure_conda
```

The command above creates three different environments including `fedbiomed-node`. You can see details in [installation tutorial](../../tutorials/installation/0-basic-software-installation.md). This documentation will focus more on configuration steps.  


**Note:** `FEDBIOMED_DIR` represents the path of the base Fed-BioMed project directory.


## Configuration Files

Configuration file is the `ini` file that is located in`{FEDBIOMED_DIR}/etc` director. It contains the variables that are required for configuring a single node. Please see the sections and the corresponding variables in a basic configuration file of a node. 

- **Default Parameters:**   
    - `id`: This is the unique ID that identifies the node. 
    - `component`: Specifies the component type. It is always `NODE` for node component
    - `version`: Version of the configuration format to avoid using older configuration files with recent Fed-BioMed versions. 

- **Researcher:**  
    - `ip`: The IP address of the researcher component that the node will connect to.
    - `port`: The port of the researcher component.   

- **Security Parameters:**
  - `hashing_algorithm`: The algorithm will be used for hashing training plan scripts to verify if the requested training plan matches the one approved on the node side.
  - `training_plan_approval`: Boolean value to switch [training plan approval](./training-plan-security-manager.md) 
  to verify training plan scripts before the training.
  - `allow_default_training_plans`: Boolean value to enable automatic approval of example training plans provided by Fed-BioMed. 
  - `secure_aggregation`: Boolean parameter (True/False) to activate secure aggregation. 
  - `force_secure_aggregation`: Boolean parameter (True/False) to force secure aggregation for every action that uses local dataset.

- **MPSDPZ (Secure Aggregation)**
  - MP-SDPZ is the library used for secure aggregation to be able to generate private/public keys securely. Please see the details [here](../secagg/configuration.md).
  - `private_key`: Path to private key to use in secure HTTP connection.
  - `public_key`: Path to public key to share with other parties (nodes and researcher) use in secure HTTP connection.
  - `mpspdz_ip`: The IP address that will be used for launching MP-SPDZ instance. 
  - `mpsdpz_port`: The port that will be used for launching MP-SPDZ instance. 
  - `allow_default_biprimes`: Boolean (True/False) to allow default biprimes for key generation. 


An example for a config file is shown below;

```ini
[default]
id = node_73768d5f-6a66-47de-8533-1291c4ef59d1
component = NODE
version = 2

[security]
hashing_algorithm = SHA256
allow_default_training_plans = True
training_plan_approval = False
secure_aggregation = True
force_secure_aggregation = False

[researcher]
ip = localhost
port = 50051

[mpspdz]
private_key = certs/cert_node_73768d5f-6a66-47de-8533-1291c4ef59d1/MPSPDZ_certificate.key
public_key = certs/cert_node_73768d5f-6a66-47de-8533-1291c4ef59d1/MPSPDZ_certificate.pem
mpspdz_ip = localhost
mpspdz_port = 14004
allow_default_biprimes = True

```

## Starting Nodes with Config Files

Currently, creating config files is done by the `fedbiomed_run` script. It automatically creates config files based on 
default values assigned in the `fedbiomed.node.environ`.  Starting nodes with specific config creates a new 
configuration file. The following command creates specific config file with default settings and starts the node. 

```
$ ./scripts/fedbiomed_run node config config-n1.ini start
```

If you run this command, you can see a new config file created in the `etc/` directory of the Fed-BioMed. 
Each node that runs in the same host should have a different node id and configuration file. Starting 
another node that uses the same config file with another raises errors. Therefore, if you launch multiple nodes please  
make sure to use different configurations.

Listing and adding datasets follows the same logic. If you want to list or add datasets in the nodes that is different 
from the default one, you need to specify the config file.

```
$ ./scripts/fedbiomed_run node config config-n1.ini list
$ ./scripts/fedbiomed_run node config config-n1.ini add
```

!!! info "Configurations for deployment"
    This is the process for the local development environment. Please see 
    [deployment instructions with VPN](../deployment/deployment-vpn.md) for production.  


## Creating configuration files for nodes
The script `${FEDBIOMEDROOT_DIR}/scripts/fedbiomed_run configuration create -c NODE -n <configuration_filename>` will create or optionally update (`-f`) a configuration file for a node with `configuration_filename` name in the `${FEDBIOMED_ROOT_DIR}/etc` folder.
More options are available and described through the help menu: `${FEDBIOMED_ROOT_DIR}/scripts/fedbiomed_run configuration create -h`
The parametrization of this script with regard to the various fields stored in the configuration happens through the usage of environment variables.
The fields that can be controlled, their associated evironment variable and default value are described as follow:

[security]:
- allow\_default\_training\_plans: ALLOW\_DEFAULT\_TRAINING\_PLANS, True
- training\_plan\_approval: ENABLE\_TRAINING\_PLAN\_APPROVAL, False
- secure\_aggregation: SECURE\_AGGREGATION, True
- force\_secure\_aggregation: FORCE\_SECURE\_AGGREGATION, False

[researcher]:
- ip: RESEARCHER\_SERVER\_HOST, ${IP\_ADDRESS}, localhost
- port: RESEARCHER\_SERVER_PORT, 50051
