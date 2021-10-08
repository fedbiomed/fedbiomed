# Fed-BioMed

version 3 implementation of the Fed-BioMed project

## Install and run in development environment

Fed-BioMed is developped under Linux Fedora, should be easily ported to other Linux distributions.

### Prerequisites :

To ensure fedbiomed will work fine, you need to install before :
* docker.
* docker-compose
* conda

### clone repo

Clone the FedBiomed repository for running the software :
```
git clone https://gitlab.inria.fr/fedbiomed/fedbiomed.git
```

Fed-BioMed developers clone of the repository :
```
git clone git@gitlab.inria.fr:fedbiomed/fedbiomed.git
```

### setup conda environments

* to create or update the environments, you use the **configure_conda** script:

```
$ ./scripts/configure_conda
```

* this script will create the environments, or update them if the conda.yaml files have been modified

* there is one specific environment for each component:

  * fedbiomed-network    : provides environement for HTTP upload/download server and MQQT daemon
  * fedbiomed-researcher : provides environement for the researcher part
  * fedbiomed-node       : provides environement for the client part

### activate the environments

In a terminal, you can configure environments to work interactively inside a specific repository, with the right conda environment and the right PYTHONPATH environment.

**WARNING**: this script only work for **bash** and **zsh**. It is not compliant with ksh/csh/tcsh/etcsh/...

```
source ./scripts/fedbiomed_environment ENV
```

where ENV chosen from:

* network
* node
* researcher

### run the software

#### run the network part

* prerequesite: **docker** must be installed and running before running the network part !!!!

* in a new terminal:

```
$ ./scripts/fedbiomed_run network
```

* this will start the necessary docker container (file repository and mqtt)

#### run the node part

* in a new terminal:

```
$ ./scripts/fedbiomed_run node start
```

* this will launch a new node

* you may also upload new data on this client with:

```
$ ./scripts/fedbiomed_run node add
```

* you may also specify a new config file for the node (usefull then running multiple test nodes on the same host)

```
$ ./scripts/fedbiomed_run node config another_config.ini start
```

* if you want to change the default IP address used to join the fedbiomed network component (localhost), you can provide it at launch time:

```
$ ./scripts/fedbiomed_run node ip_address 192.168.0.100 start
$ ./scripts/fedbiomed_run researcher ip_address 192.168
```

(adjust the 192.168.0.100 IP address to your configuration)

If this option is given at the first launch or after a clean, it is saved in the configuration file and becomes the default for subsequent launches. If this option is given at a subsequent launch, it only affects this launch.

#### run a researcher notebook

* in a new terminal:

```
$ ./scripts/fedbiomed_run researcher
```

* this will launch a new jupyter notebook working in the **notebooks** repository. Some notebooks are available:

  - `getting-started.ipynb` : training a simplenet + federated average on MNIST data
  - `local_training.ipynb` : comparing the simplenet + federated average on MNIST data with its local training equivalent


#### run a researcher script

* in a new terminal:

```
$ source ./scripts/fedbiomed_environment researcher
```

* then you can use any researcher script

```
$ python ./notebooks/getting-started.py
```

### change IP address for network in the current bash

By default, fedbiomed-{node,researcher} contact fedbiomed-network on `localhost`.
To configure your current shell to use another IP address for joining fedbiomed-network (e.g. 192.168.0.100):

```bash
source ./scripts/fedbiomed_environment network
source ./scripts/fedbiomed_environment node 192.168.0.100
source ./scripts/fedbiomed_environment researcher 192.168.0.100
```

Then launch the components with usual commands while you are in the current shell.

Warning: this option does not modify the existing configuration file (.ini file).


This currently doesn't support scenario where node and researcher do not use the same IP address to contact the network (eg: NAT for one component).


### clean state (restore environments back to new)

De-configure environments, remove all configuration files and caches
```
source ./scripts/fedbiomed_environment clean
```



## Misc developper tools to help debugging

### lqueue

list the content of a message queue (as used in fedbiomed.node and fedbiomed.researcher)

usage:  lqueue directory
   or
        lqueue dir1 dir2 dir3 ...

## Jupyterhub 

### Build Containers and run

To build and your jupyterhub condatiners run following command in base directory

`docker-compose up --build`

Open your browser and go http://localhost:8000/fedbiomed. Login with follwing username and password.

User: fedbiomed
pass: FED123fed