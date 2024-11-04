---
title: Node Configuration
description: Configuration instructions for Fed-BioMed nodes.
keywords: fedbiomed configuration,node configuration
---

# Node Configuration

Fed-BioMed framework has 2 main components: `node` and `researcher`. A `node` stores private datasets and
performs training in response to researcher's train requests. It communicates directly with the `researcher` component using RPC protocol.


A basic `node` component configuration contains the following settings:

- a python environment
- a unique node id
- security parameters such as training plan approval mode, hashing algorithms
- connection credentials for the `researcher` component

!!! note "Note"
    These basic configurations are created automatically using default values.
    While it is possible to manually edit the configuration files, some parameters may become incompatible upon doing so; such as [server]/host and [server]/pem since the latter depends on the former.
    It is therefore strongly advised to rely on the dedicated script for configuration creation and refresh, namely `fedbiomed_run configuration create` whose options are described above.

## Environment for Nodes

A `node` requires a conda environment to be able to run.
This environment provides the necessary python modules for both the task management part and the model training part.
Thanks to Fed-BioMed's Node CLI, this conda environment can be created easily.

```
$ ${FEDBIOMED_DIR}/scripts/configure_conda node
```

The command above creates the `fedbiomed-node` conda environment.
You can find more details in [installation tutorial](../../tutorials/installation/0-basic-software-installation.md).

**Note:** `FEDBIOMED_DIR` represents the path of the base Fed-BioMed project directory.


## Configuration Files

A configuration file is an `ini` file that is located in the `${FEDBIOMED_DIR}/etc` directory.
Each configuration file is unique to one node.

The configuration file is structured following the sections below:

- **Default Parameters:**
    - `id`: This is the unique ID that identifies the node.
    - `component`: Specifies the component type. It is always `NODE` for node component
    - `version`: Version of the configuration format to avoid using older configuration files with recent Fed-BioMed versions.
    - `db`: Relative path to the node's database, from this configuration file

- **Authentication and Identification**
    - `private_key`: Path to private key to use identify the component.
    - `public_key`: Path to public key to share with other parties (nodes and researcher) to verify its identify.

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
    - `secagg_insecure_validation`: A boolean parameter to activate insecure validation for secure aggregation rounds to verify the correctness of the aggregation on the researcher side. This option is intended for testing and debugging purposes and must be deactivated in production deployments.

An example for a config file is shown below;

```ini
[default]
id = node_73768d5f-6a66-47de-8533-1291c4ef59d1
component = NODE
version = 2
db = ../var/db_NODE_94572b0f-f55c-4167-b729-f16247c35a04.json

[security]
hashing_algorithm = SHA256
allow_default_training_plans = True
training_plan_approval = False
secure_aggregation = True
force_secure_aggregation = False
secagg_insecure_validation = False


[researcher]
ip = localhost
port = 50051

```

## Starting Nodes with Config Files

Starting nodes with specific config creates a new configuration file.
The following command creates a specific config file with default settings and starts the node.

```
$ ./scripts/fedbiomed_run node --config config-n1.ini start
```

If you run this command, you can see a new config file created in the `etc/` directory of the Fed-BioMed.
Each node that runs in the same host should have a different node id and configuration file. Starting
another node that uses the same config file does not raise an error message but results in errors during training.
Therefore, if you launch multiple nodes please make sure to use different configurations.

The `--config` flag can be used with any of the `node` subcommands as in the examples below.

```
$ ./scripts/fedbiomed_run node --config config-n1.ini dataset list
$ ./scripts/fedbiomed_run node --config config-n1.ini dataset add
```

!!! info "Configurations for deployment"
    The process described above is valid for the local development environment. Please see
    [deployment instructions with VPN](../deployment/deployment-vpn.md) for production.

## Creating advanced configuration files for `node` components

The command below can be used to create, or optionally recreate (`-f`), a configuration file for a node with
`CONFIGURATION_FILE_NAME` name.

```
${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create -c NODE -n [CONFIGURATION_FILE_NAME]
```

You may view and edit your new configuration file at the location `${FEDBIOMED_DIR}/etc/[CONFIGURATION_FILE_NAME]`.

!!! warning "Recreating a configuration destroys the node id"
    Recreating (`-f`) a configuration file overrides the whole file including the node id.
    To preserve the node id across updates of the configuration file, prefer using `${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration refresh -c NODE -n node_1.ini`.

Environment variables can be used to parametrize the various options for creating a configuration file. The fields that can be controlled, their associated environment variable, and default value are described below:

```
[security]:
- allow_default_training_plans: ALLOW_DEFAULT_TRAINING_PLANS, True
- training_plan_approval: ENABLE_TRAINING_PLAN_APPROVAL, False
- secure_aggregation: SECURE_AGGREGATION, True
- force_secure_aggregation: FORCE_SECURE_AGGREGATION, False
- secagg_insecure_validation: SECAGG_INSECURE_VALIDATION, True

[researcher]:
- ip: RESEARCHER_SERVER_HOST, ${IP_ADDRESS}, if not set: localhost
- port: RESEARCHER_SERVER_PORT, 50051
```

You may also explore them through the command's help menu

```
${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create -h
```

### Examples

Create the new configuration file `${FEDBIOMED_DIR}/etc/node_1.ini` with a specified IP and port for the researcher,
and disabling secure aggregation. Forcefully delete and recreate the file if it was already existing.

```
$ RESEARCHER_SERVER_HOST=121.203.21.147 RESEARCHER_SERVER_PORT=8909 SECURE_AGGREGATION=0 ${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create -c NODE -n node_1.ini -f
```
