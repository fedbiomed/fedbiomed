
# Configuring and Using the Researcher Component in Fed-BioMed

The Researcher component in Fed-BioMed serves as the central orchestrator for collaborative learning experiments. It allows users to define training plans, configure experiments, and manage communication with multiple nodes. This guide explains how to configure and use the Researcher component in detail.

---

## Creating the Researcher Component

To begin, create a Researcher component in your working directory. The following command initializes a new Researcher component:

```shell
fedbiomed component create -c researcher
```

This command creates a folder named `fbm-researcher` in your current working directory. This folder contains all the necessary assets for the Researcher component, including configuration files, experiment assets and logs.

It is also possible to create the Researcher component in a custom directory using the `--path` option:

```shell
fedbiomed component create -c researcher --path /path/to/custom-directory
```

---

##  Starting the Researcher Component

Executing the following command will start the Researcher component.

```shell
fedbiomed researcher start
```

This command launches a Jupyter Notebook server preconfigured to use the Researcher component created in your working directory. If the Researcher component is located in a different directory, the path should be specified using the `--path` option:

```shell
fedbiomed researcher --path /path/to/fbm-researcher start
```

After the Jupyter Notebook is launched, it serves as an interface for defining [training plans](./training-plan.md) and [experiments](./experiment.md). Preconfigured notebooks are available in the [tutorials](../../tutorials/pytorch/index.md) to assist with these tasks.

---

## Configuring the Researcher Component

The Researcher component’s configuration file is located in the `etc` folder of the `fbm-researcher` directory. For example:

```
/path/to/fbm-researcher/etc/config.ini
```

Here is an example configuration file:
```
[default]
id = RESEARCHER_2ba562cc-6943-4430-8f79-cb3877b2ea79
component = RESEARCHER
version = 3.1.0
db = ../var/db_RESEARCHER_2ba562cc-6943-4430-8f79-cb3877b2ea79.json

[server]
host = localhost
port = 50051
node_disconnection_timeout = 10

[certificate]
private_key = certs/server_certificate.key
public_key = certs/server_certificate.pem

[security]
secagg_insecure_validation = True
```

This file contains settings for connecting to a server, managing security certificates, and configuring the database. Here's a breakdown of each section:

### `[default]`
- **id**: This is a unique identifier for the researcher.
- **component**: Refers to the component or role being used, here it's `RESEARCHER`, indicating that the configuration is for a researcher component.
- **version**: The version of the configuration.
- **db**: Specifies the location of the database used by this researcher component, pointing to a JSON file. This file likely stores data related to the researcher’s activities or state.

### `[server]`
- **host**: Defines the server host address to connect to.
- **port**: Specifies the port number `50051` for the server connection.
- **node_disconneciton_timeout**: A node is considered disconnected if it does not subscribe to new tasks within a given number of seconds.

### `[certificate]`
- **private_key**: The path to the server's private key, which is used for secure communication.
- **public_key**: The path to the server's public key, stored at `certs/server_certificate.pem`, which is used for encryption in secure communications.

### `[security]`
- **secagg_insecure_validation**: Set to `True`, this indicates that the system is using insecure validation for secure aggregation. This is used for development purposes, but should be switched to `False` for production to ensure proper security.

To modify the configuration, open the file in a text editor and make the necessary changes. Ensure that the component is restarted after editing the configuration file to apply the changes. This restart can be done by relaunching researcher component, restarting Jupyter Notebook or re-executing a python script that defines an experiment.


---


## Using Plain Python Scripts Without Jupyter Notebook

If you prefer to use plain Python scripts instead of the Jupyter Notebook interface, ensure that the environment variable `FBM_RESEARCHER_COMPONENT_ROOT` points to the correct Researcher component directory:

### Option 1: Set the Environment Variable in Your Script

```python
import os
os.environ['FBM_RESEARCHER_COMPONENT_ROOT'] = '/path/to/fbm-researcher'

# Import and use Fed-BioMed components
```

### Option 2: Export the Variable in the Shell

```shell
export FBM_RESEARCHER_COMPONENT_ROOT=/path/to/fbm-researcher
python my_script.py
```

### Option 3: Inline Environment Variable

```shell
FBM_RESEARCHER_COMPONENT_ROOT=/path/to/fbm-researcher python my_script.py
```

---

##  Managing Multiple Researcher Configurations

While the default Researcher component is named `fbm-researcher`, it is always possible to create multiple Researcher components for different configurations. To switch between configurations, specify the path to the desired Researcher component when starting or using it:

```shell
fedbiomed researcher --path /path/to/another-researcher start
```

---

## Troubleshooting

1. **Component Not Found or Duplicated:** Ensure that the `FBM_RESEARCHER_COMPONENT_ROOT` environment variable or the `--path` option points to the correct directory. If researcher environment is going to be initialized for the for time the component directory should be empty.
2. **Communication Errors:** Verify the IP address, port, and certificates in the configuration file.
3. **Permission Denied:** Check that you have write permissions for the working directory.


### Running Researcher with delayed Network and Heavy ML Models

In real-world applications, communication delays can be significant, causing default settings to fail. One encountered issue is that nodes may be considered disconnected if they do not re-subscribe to tasks from the researcher within a certain time frame. By default, the researcher expects nodes to send a new task subscription request within 10 seconds. However, due to network latency, message processing time, or other delays, node reconnection or subscription may take longer than 10 seconds.

To address this, you can adjust the timeout setting on the researcher server using the `node_disconnection_timeout` configuration parameter.


