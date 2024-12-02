
# Configuring and Using the Researcher Component in Fed-BioMed

The Researcher component in Fed-BioMed serves as the central orchestrator for collaborative learning experiments. It allows users to define training plans, configure experiments, and manage communication with multiple nodes. This guide explains how to configure and use the Researcher component in detail.

---

## Creating the Researcher Component

To begin, create a Researcher component in your working directory. The following command initializes a new Researcher component:

```shell
fedbiomed component create -c researcher
```

This command creates a folder named `fbm-researcher` in your current working directory. This folder contains all the necessary assets for the Researcher component, including configuration files and logs.

If you wish to create the Researcher component in a custom directory, use the `--path` option:

```shell
fedbiomed component create -c researcher --path /path/to/custom-directory
```

---

##  Starting the Researcher Component

To start the Researcher component, execute the following command:

```shell
fedbiomed researcher start
```

This command launches a Jupyter Notebook server preconfigured to use the Researcher component created in your working directory. If the Researcher component is located in a different directory, specify the path using the `--path` option:

```shell
fedbiomed researcher --path /path/to/fbm-researcher start
```

Once the Jupyter Notebook is launched, you can use it to define [training plans]() and [experiments]().

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
version = 3
db = ../var/db_RESEARCHER_2ba562cc-6943-4430-8f79-cb3877b2ea79.json

[server]
host = localhost
port = 50051

[certificate]
private_key = certs/server_certificate.key
public_key = certs/server_certificate.pem

[security]
secagg_insecure_validation = True
```

This file contains includes settings for connecting to a server, managing security certificates, and configuring the database. Here's a breakdown of each section:

### `[default]`
- **id**: This is a unique identifier for the researcher.
- **component**: Refers to the component or role being used, here it's `RESEARCHER`, indicating that the configuration is for a researcher component.
- **version**: The version of the configuration.
- **db**: Specifies the location of the database used by this researcher component, pointing to a JSON file. This file likely stores data related to the researcher’s activities or state.

### `[server]`
- **host**: Defines the server host address to connect to.
- **port**: Specifies the port number `50051` for the server connection.
### `[certificate]`
- **private_key**: The path to the server's private key, which is used for secure communication.
- **public_key**: The path to the server's public key, stored at `certs/server_certificate.pem`, which is used for encryption in secure communications.

### `[security]`
- **secagg_insecure_validation**: Set to `True`, this indicates that the system is using insecure validation for secure aggregation. This is used for development purposes, but should be switched to `False` for production to ensure proper security.

To modify the configuration, open the file in a text editor and make the necessary changes. Ensure that the component is restarted after editing the configuration file to apply the changes:



```shell
fedbiomed researcher restart
```

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

While the default Researcher component is named `fbm-researcher`, you can create multiple Researcher components for different configurations. To switch between configurations, specify the path to the desired Researcher component when starting or using it:

```shell
fedbiomed researcher --path /path/to/another-researcher start
```

---

## Troubleshooting

1. **Component Not Found or Duplicated:** Ensure that the `FBM_RESEARCHER_COMPONENT_ROOT` environment variable or the `--path` option points to the correct directory.
2. **Communication Errors:** Verify the IP address, port, and certificates in the configuration file.
3. **Permission Denied:** Check that you have write permissions for the working directory.



