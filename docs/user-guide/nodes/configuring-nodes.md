---
title: Node Configuration
description: Configuration instructions for Fed-BioMed nodes.
keywords: fedbiomed configuration,node configuration
---

# Node Configuration


The Fed-BioMed framework has two main components: `node` and `researcher`. A `node` stores private datasets and performs training in response to the `researcher's` training requests. Communication between the `node` and the `researcher` component occurs via the RPC protocol.

### Basic `node` Configuration

A basic configuration for the `node` component includes the following settings:

- A Python environment
- A unique node ID
- Security parameters, such as training plan approval mode and hashing algorithms
- Connection credentials for communicating with the `researcher` component

!!! note "Note"
    These basic configurations are generated automatically with default values. While it is possible to manually edit the configuration files, doing so may render certain parameters incompatible. For example, `[server]/host` and `[server]/pem` must remain consistent, as the latter depends on the former.
    To avoid such issues, it is strongly recommended to use the dedicated script for creating and refreshing configurations:
    **`fedbiomed component create`**, with options described above.


## Environment for Nodes

Fed-BioMed can be installed using `pip`. It can be installed into native host machine or on virtual environment using tools such as `pyenv`, `conda` or `virtualenv`.  You can find more details in [installation tutorial](../../getting-started/installation.md).


## Node Component Installation

Fed-BioMed uses the default `fbm-node` directory to store all component-related files, including temporary files, variables, configuration files, certificates, and other essential documents.

When a basic command is executed, Fed-BioMed will automatically create a default `fbm-node` directory if it does not already exist. If the directory is already present, the system will use the existing one. The component directory is created relative to the directory where the Fed-BioMed commands are executed. For more details, please refer to the generic [Component Creation Guide](../../getting-started/configuration.md).

For example, the following sequence of commands will create a directory named `my-workdir` in the home directory, navigate to it, and then start the Fed-BioMed node. This operation will automatically instantiate the node component within the `my-workdir` directory, creating a folder named `fbm-node`, which serves as the default directory for the Fed-BioMed node component.   Since `fbm-node` does not yet exist, the CLI will prompt you for permission to initialize the component.

```bash
mkdir $HOME/my-workdir
cd $HOME/my-workdir
fedbiomed node start
```

In another terminal, you can inspect the folder structure using the following command:

```bash
tree $HOME/my-workdir/fbm-node
```

The structure will resemble the tree below:

```plaintext
fbm-node
├── etc
│   ├── certs
│   │   ├── FBM_certificate.key
│   │   └── FBM_certificate.pem
│   └── config.ini
└── var
    ├── cache
    ├── db_NODE_992aab29-c485-463f-a84b-6ce055c17c2e.json
    ├── tmp
    └── training_plans
```

Please see section [Using non-default Fed-BioMed node](#using-non-default-Fed-BioMed-node) to select different node installations.

### Configuration file: config.ini

A configuration file is an `ini` file that is located in the `${FEDBIOMED_DIR}/etc` directory. Each configuration file is unique to one node.

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

## Using non-default Fed-BioMed node

The option `--path` that comes after specification `node` in Fed-BioMed command allows to use different node component in different directories than the default one. `--path` can be relative or absolute. As it is in default command execution, if the given directory for node is existing it will created automatically.

```
$ fedbiomed node --path my-node start
```
The command above will create a folder called `my-node` in the directory where the command is executed, or use the one already existing to start a node.

The `--path` flag can be used with any of the `node` subcommands as in the examples below.

```
$ fedbiomed node --path my-node dataset list
$ fedbiomed node --path my-node dataset add
```

## Creating a component without doing any action


The `component create` action allows you to initialize a Fed-BioMed component without needing to start it or add a dataset to a node. This option is ideal for initializing a node component and configuring it later, such as making changes to the configuration file.

```shell
fedbiomed component create -c NODE --path </path/for/component>
```

The command will raise an error if a component has already been initialized in the specified `-r` root directory. To prevent this error, you can use the `--exist-ok` option.

Once the component is initialized, you can edit its configuration file located at `path/to/component/etc/config.ini`.

---

## Environment Variables

Environment variables can be used to parameterize various options for creating the configuration file. The fields that can be controlled, along with their associated environment variables and default values, are described below:

[security]:
- allow_default_training_plans: FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS, True
- training_plan_approval: FBM_SECURITY_TRAINING_PLAN_APPROVAL, False
- secure_aggregation: FBM_SECURITY_SECURE_AGGREGATION, True
- force_secure_aggregation: FBM_SECURITY_FORCE_SECURE_AGGREGATION, False
- secagg_insecure_validation: FBM_SECURITY_SECAGG_INSECURE_VALIDATION, True

[researcher]:
- ip: FBM_RESEARCHER_IP, ${IP_ADDRESS}, if not set: localhost
- port: FBM_RESEARCHER_PORT, 50051
```

### Examples

Create the new component with a specified IP and port for the researcher, and disabling secure aggregation. Forcefully delete and recreate the file if it was already existing.

```shell
export FBM_RESEARCHER_IP=121.203.21.147
export FBM_RESEARCHER_PORT=8909
export FBM_SECURITY_SECURE_AGGREGATION=0
fedbiomed component create -c NODE -p my-node
```

