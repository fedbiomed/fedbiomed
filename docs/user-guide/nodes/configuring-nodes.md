---
title: Node Configuration
description: Configuration instructions for Fed-BioMed nodes. 
keywords: fedbiomed configuration,node configuration
---

# Node Configuration

Fed-BioMed framework has 3 main components: `node`, `network`, and `researcher`. A `node` stores private datasets and 
perform training in response to researcher's train requests. It communicates with the `researcher` component over MQTT
(messaging server) that is running in the `network` component. It also uses file repository of the `network` component 
to exchange model parameters and files with the researcher. 

A basic node configuration contains following settings;

- providing a python environment
- assigning a unique node id 
- setting security parameters such as training plan approval mode, hashing algorithms 
- providing the connection credentials for the network component

!!! note "Note"
    These basic configurations are done automatically using default values by the scripts provided in Fed-FedBioMed. 
    However, it is possible to modify the configuration manually through configuration file. 

## Environment for Nodes 

A `node` requires a conda environment to be able to run. This environment provides necessary python modules for both the task management part and the model training part.  Thanks to Fed-BioMed Node CLI, this conda environment can be created easily. 

```
$ ${FEDBIOMED_DIR}/scripts/configure_conda
```

The command above creates three different environments including `fedbiomed-node`. You can see details in [installation tutorial](../../tutorials/installation/0-basic-software-installation.md). This documentation will focus more on configuration steps.  


**Note:** `FEDBIOMED_DIR` represents the path of the base Fed-BioMed project directory.


## Config Files

Config files are the `ini` files including the following information.

- **Default Parameters:**   
    - `node_id`: This is the unique id which identifies the node. 
    - `uploads_url`: The URL string that indicates upload request URL for the model parameters. 

- **MQTT Parameters:**  
    - `broker_ip`: The IP address for connecting MQTT to consume and publish messages with the researcher
    - `port`: Connection port for MQTT messaging server
    - `keep_alive`: Delay in seconds before sending an applicative MQTT ping if there is no MQTT exchange during this period.   

- **Security Parameters:**
  - `hashing_algorithm`: The algorithm will be used for hashing training plan scripts to verify if the requestes 
  training plan matches the one approved on the node side.
  - `training_plan_approval`: Boolean value to switch [training plan approval](./training-plan-security-manager.md) 
  to verify training plan scripts before the training.
  - `allow_default_training_plans`: Boolean value to enable automatic approval of example training plans provided by Fed-BioMed. 
  

An example for a config file is shown below;

```
[default]
node_id = node_7fd39224-4040-448f-8360-577e2066e2ce
uploads_url = http://localhost:8844/upload/

[mqtt]
broker_ip = localhost
port = 1883
keep_alive = 60

[security]
hashing_algorithm = SHA256
allow_default_training_plans = True
training_plan_approval = False


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

