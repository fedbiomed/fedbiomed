# Fed-BioMed

version 3 implementation of the Fed-BioMed project

## Introduction

Fed-BioMed is an open source project focused on empowering biomedical research using non-centralized approaches for statistical analysis and machine learning.

The project is currently based on Python, PyTorch and Scikit-learn, and enables developing and deploying federated learning analysis in real-world machine learning applications.

The code is regularly released and available on the **master** branch of this repository. The documentation of the releases can be found at https://fedbiomed.gitlabpages.inria.fr

Curious users may also be interested by the current developments, occuring in the **develop** branch (https://gitlab.inria.fr/fedbiomed/fedbiomed/-/tree/develop)
According to our coding rules, the develop branch is usable, tests and tutorials will run, but the documentation may be not fully available or desynchronized with the code. We only provide support for the last release (currently v3.4) aka the master branch.


## Install and run in development environment

Fed-BioMed is developped under Linux Fedora, should be easily ported to other Linux distributions.
It runs also smoothly on macOSX (mostly tested on macOSX 12: Monterey).

This README.md file provide a quick start/installation guide for Linux.

Full installation instruction are also available at: https://fedbiomed.gitlabpages.inria.fr/v3.4/tutorials/installation/0-basic-software-installation/
An installation guide is also provided for Windows10, which relies on WSL2: https://fedbiomed.gitlabpages.inria.fr/v3.4/user-guide/installation/windows-installation/


### Prerequisites :

To ensure fedbiomed will work fine, you need to install before :

* docker
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

* to create or update the environments, you can use the **configure_conda** script:

```
$ ./scripts/configure_conda
```

* this script will create/upadte the conda environments

* there is one specific environment for each component:

  * fedbiomed-network.yaml    : environment for HTTP upload/download server and MQQT daemon (network component)
  * fedbiomed-node.yaml       : environment for the node part
  * fedbiomed-researcher.yaml : environment for the researcher part
  * fedbiomed-gui.yaml        : environment for the data management gui on the node

**Remark**:

* this script can also be used to update only some of the environments
* for some components, we provide different versions of yaml files depending of the operating system of your host
* in case of (conda or python) errors, we advice to remove all environments and restart from fresh (use the **-c** flag of configure_conda)
* general usage for this script is:

```
Usage: configure_conda [-n] [-c] [-t] [ENV ENV ..]

Install/update conda environments for fedbiomed. If several ENV
are provided, only these components will be updated. If no ENV is
provided, all components will be updated.

ENV can be network, node, researcher, gui (or a combination of them)

 -h, --help            this help
 -n, --dry-run         do nothing, just print what the script would do
 -c, --clean           remove environment before reinstallating it
 -t, --test            test the environment at the end of installation
                       (this only tests the researcher environment for now)
```


### activate the environments

In a terminal, you can configure environments to work interactively inside a specific repository, with the right conda environment and the right PYTHONPATH environment.

**WARNING**: this script only work for **bash**, **ksh** and **zsh**. It is not compliant with c variant of shell (csh/tcsh/etcsh/...)

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

* you may also upload new data on this node with:

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

  - `101_getting-started.ipynb` : training a simplenet + federated average on MNIST data
  - `pytorch-local-training.ipynb` : comparing the simplenet + federated average on MNIST data with its local training equivalent


#### run a researcher script

* in a new terminal:

```
$ source ./scripts/fedbiomed_environment researcher
```

* then you can use any researcher script

```
$ python ./notebooks/101_getting-started.py
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

## Install and run in vpn+development environment

### Files

The **envs/vpn** directory contains all material for VPN support.
A full technical description is provided in **envs/vpn/README_tempo.md**

The **./scripts/fedbiomed_vpn** script is provided to ease the deployment of
a set of docker container(s) with VPN support. The provided containers are:

- fedbiomed/vpn-vpnserver: WireGuard server
- fedbiomed/vpn-restful: HTTP REST communication server
- fedbiomed/vpn-mqtt: MQTT message broker server
- fedbiomed/vpn-researcher: a researcher jupyter notebooks
- fedbiomed/vpn-node: a node component
- fedbiomed/vpn-gui: a GUI for managing node component data

All these containers are communicating through the Wireguard VPN server.

### Setup and run all the docker containers

To setup **all** these components, you should:

- clean all containers and files

```
./scripts/fedbiomed_vpn clean
```

- build all the docker containers

```
./scripts/fedbiomed_vpn build
```

- configure the wireguard encryption keys of all containers

```
./scripts/fedbiomed_vpn configure
```

- start the containers

```
./scripts/fedbiomed_vpn start
```

- check the containers status (presence and Wireguard configuration)

```
./scripts/fedbiomed_vpn status
```

- run a **fedbiomed_run** command inside the node component. Eg:

```
./scripts/fedbiomed_vpn node --add-mnist /data
./scripts/fedbiomed_vpn node list
./scripts/fedbiomed_vpn node start
```

- connect to the researcher jupyter at http://127.0.0.1:8888
(Remark: the *researcher** docker automatically starts a jupyter notebook inside the container)

- manage data inside the node at http://127.0.0.1:8484

- stop the containers:

```
./scripts/fedbiomed_vpn stop
```

### managing individual containers

You can manage individually the containers for the build/stop/start phases,
by passing the name of the container(s) on the command line.

For example, to build only the node, you can use:

```
./scripts/fedbiomed_vpn build node
```

You can build/configure/stop/start/check more than one component at a time. Example:
```
./scripts/fedbiomed_vpn build gui node
```

This will stop and build the node container.

The list of the container names is:

- vpnserver
- mqtt
- restful
- researcher
- node
- gui

**Remarks**:
- the configuration files are keeped then rebuilding individual containers
- to remove the ald config files, you should do a **clean**
- restarting only network component (vpnserver, restful, mqqt) then others are running
may lead to unpredictable behavior. In this case, it is adviced to restart from scratch
(clean/build/configure/start)



## Misc developper tools to help debugging

### scripts/lqueue

list the content of a message queue (as used in fedbiomed.node and fedbiomed.researcher)

usage:  lqueue directory
   or
        lqueue dir1 dir2 dir3 ...

### scripts/run\_integration\_test

Run a full (integration) test by launching:

- a researcher (running a python script or a notebook script)
- several nodes, providing data
- the network component.

Usefull for continous integration tests and notebook debugging.
Full documentation in tests/README.md file.


## Using Tensorboard

To enable tensorboard during traning routine to see loss values, you need to set `tensorboard` parameter to `true` while initializing Experiment class.

```
exp = Experiment(tags=tags,
                 #nodes=None,
                 model_path=model_file,
                 model_args=model_args,
                 model_class='MyTrainingPlan',
                 training_args=training_args,
                 round_limit=round_limit,
                 aggregator=FedAverage(),
                 node_selection_strategy=None,
                 tensorboard=True
                )
```
Or after initialization :
```
exp.set_tensorboard(True)
```

During training, the scalar values (loss) will be writen in the `runs` directory. You can either start tensorboard from jupyter notebook or terminal window.

**Start tensorboard from notebook**

First you should import `TENSORBOARD_RESULTS_DIR` from researcher environment in another cell

```python
from fedbiomed.researcher.environ import environ
tensorboard_dir = environ['TENSORBOARD_RESULTS_DIR']
```

Load tensorboard extension in a different code block.

```python
%load_ext tensorboard
```

Run following command to start tensorboard

```python
tensorboard --logdir "$tensorboard_dir"
```

**Start tensorboard from terminal command line**

Open new terminal and change directory to Fed-BioMed base directory (`${FEDBIOMED_DIR}`)

Make sure that already activated fedbiomed researcher conda environment :

```bash
source ./scripts/fedbiomed_environment researcher
```

Launch tensorboard with the following command :

```bash
tensorboard --logdir "$tensorboard_dir"`
```


## Model Hashing and Enabling Model Approve

Fed-BioMed offers optional model approval feature to approve the models requested by the researcher. This model approval process is done by hashing/checksum operation by the `ModelManager` of node instance. When the `MODEL_APPROVE` mode is enabled, node should register/approve model files before performing the training. For testing and easy development, there are already presented default models by Fed-BioMed for the tutorials that we provide in the `notebooks` directory. However, node can also enable or disable the mode for allowing default models to perform training.

#### Config file for security parameters

Enabling model approval mode, allowing default Fed-BioMed models and the hashing algorithm that will be performed for the checksum operation can be configurred from the config file of the node. The following code snippet represents an example security section of config file with default values.

```
[default]
# ....

[mqtt]
# ....

[security]
hashing_algorithm = SHA256
allow_default_models = True
model_approval = False

```

By default, when node is launched for the first time without additional security parameters, `model_approval` mode comes as disabled. If `model_approval` is disabled the status of `allow_defaults_models` will have no effect. To enable `model_approval` you should set `model_approval` to `True` and if it is desired `allow_default_models` can be set to `False` for not accepting models of default Fed-BioMed examples.

The default hashing algorithm is `SHA256` and it can also be changed to other hashing algorithms that are provided by Fed-BioMed. You can see the list of Hashing algorithms in the following section.


#### Hashing Algorithms

`ModelManager` provides different hashing algorithms, and the algorithm can be changed through the config file of the node. The name of the algorithms should be typed with capital letters. However, after changing hashing algorithm node should be restarted because it checks/updates hashing algorithms of the register/default models during the starting process.

Provided hashing algorithms are `SHA256`, `SHA384`, `SHA512`, `SHA3_256`, `SHA3_384`, `SHA3_512`, `BLAKE2B` and `BLAKE2S`. These are the algorithms that has been guaranteed by `hashlib` library of Python.


#### Starting nodes with different modes

To enable `model_approval` mode and `allow_default_models` node, start the following command.

```shell
./scripts/fedbiomed_run node config config-n1.ini --enable-model-approval --allow-default-models start
```

This command will start the node with model approval activated mode even the config file has been set as `model_aprove = False`. However it doesn't change the config file. If there is no config file named `config-n1.ini` it creates a config file for the node with enabled model approval mode.

```
[security]
hashing_algorithm = SHA256
allow_default_models = True
model_approval = True


For starting node with disabled model approval and default models;

```shell
./scripts/fedbiomed_run node config config-n1.ini --disable-model-approval --disable-default-models start
```

#### Default Models

Default models are located in the `envs/common/default_models/` directory as `txt` files. Each time  the node starts with the `model_approval = True` and `allow_default_model = True` modes, hashing of the model files are checked to detect if the file is modified, the hashing algorithm has changed or is there any new model file added. If model files are modified `ModelManager` updates hashes for these models in the database. If the hashing algorithm of the model is different from the active hashing algorithm, hashes also get updated. This process only occurs when both `model-approval` and `allow-default-models` modes are activated. To add new default model for the examples or for testing, model files should be saved as `txt` and copied into the `envs/common/default_models` directory. After the copy/save operation node should be restarted.


#### Registering New Models

New models can be registered using `fedbiomed_run` scripts with `register-model` option.

```shell
./scripts/fedbiomed_run node config config-n1.ini register-model
```

The CLI asks for the name of the model, description and the path where model file is stored. **Model files should be saved as txt in the file system for registration**. This is because these files are for only hashing purposes not for loading modules.

#### Deleting Registered Models

Following command is used for deleting registered models.

```
./scripts/fedbiomed_run node config config-n1.ini delete-model
```

Output of this command will list registered models with their name and id. It will ask to select model file you would like to remove. For example, in the follwing example, typing `1`  will remove the `MyModel` from registered/approved list of models.

```
Select the model to delete:
1) MyModel	 Model ID model_98a1e68d-7938-4889-bc46-357e4ce8b6b5
Select:
```

Default models can not be removed using fedbiomed CLI. They should be removed from the `envs/common/default_models` directory. After restarting the node, deleted model files will be also removed from the `Models` table of the node DB.


#### Updating Registered model

Following command is used for updating registered models. It updates chosen model with provided new model file. User also
can provide same model file to update its content.

```
./scripts/fedbiomed_run node config config-n1.ini update-model
```

## Fed-BioMed Node GUI

Node GUI provides an interface for Node to manage datasets and deploy new ones. GUI consists of two components, Server and UI. Server is developed on Flask framework and UI is developed using ReactJS. Flask provides API
services that use Fed-BioMed's DataManager for deploying and managing dataset. All the source files for GUI has been
located on the `${FEDBIOMED_DIR}/gui` directory.

### Starting GUI

Node GUI can be started using Fed-BioMed CLI.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run gui data-folder '<path-for-data-folder>' config '<name-of-the-config-file>' start
```

Arguments:

- ``data-folder``: Data folder represents the folder path where datasets have been stored. It can be absolute or relative path.
If it is relative path, Fed-BioMed base directory is going to be used as reference. **If `datafolder` is not provided. Script will look for
`data` folder in the Fed-BioMed root directory and if it doesn't exist it will raise an error.**
- ``config``: Config file represents the name of the configuration file which is going to be used for GUI. If it is not
provided, default will be``config_node.ini``.

It is also possible to start GUI on specific host and port, By default it is started `localhost` as host and `8484` as port.  To change
it you can modify following command.

**IMPORTANT:** Please always consider providing `data-folder` argument while starting the GUI.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run gui data-folder ../data config config-n1.ini --port 80 --host 0.0.0.0 start

```

### Details of Start Process

When the Node GUI is started, it installs `npm` modules and builds ReactJS application in ``${FEDBIOMED_DIR}/var/gui-build``. If the GUI
is already built (means that `gui/ui/node_modules` and `var/gui-build` folders exist), it does not reinstall and rebuild ReactJS. If you want to
reinstall and rebuild, please add `--recreate` flag in the command same as below,

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run gui data-folder ../data --recreate start
```


### Launching Multiple Node GUI

It is possible to start multiple Node GUIs for different nodes as long as the http ports are different. The
commands below starts three Node GUI for the nodes; config-n1.ini, config-n2.ini and config-n3.ini on the ports respectively, `8181`, `8282` and `8383`.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run data-folder ../data gui config config-n1.ini port 8181 start
${FEDBIOMED_DIR}/scripts/fedbiomed_run data-folder ../data gui config config-n2.ini port 8282 start
${FEDBIOMED_DIR}/scripts/fedbiomed_run data-folder ../data gui config config-n3.ini port 8383 start
```

### Development/Debugging for GUI

If you want to customize or work on user interface for debugging purposes, it is always better to use ReactJS in development mode, otherwise building GUI
after every update will take a lot of time. To launch user interface in development mode first you need to start Flask server. This can be
easily done with the previous start command. Currently, Flask server always get started on development mode.  To enable debug mode you should add `--debug`
flag to the start command.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run gui data-folder ../data config config-n1.ini --debug start
```
**Important:** Please do not change Flask port and host while starting it for development purposes. Because React (UI) will be calling
``localhost:8484/api`` endpoint in development mode.

The command above will serve ``var/gui-build`` directory as well as API services. It means that on the URL `localhost:8484` you will be able to
see the user interface. This user interface won't be updated automatically because it is already built. To have dynamic update for user interface you can start React with ``npm start``.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_environment gui
cd ${FEDBIOMED_DIR}/gui/ui
npm start
```

After that if you go ``localhost:3000`` you will see same user interface is up and running for development.  When you change the source codes
in ``${FEDBIOMED_DIR}/gui/ui/src`` it will get dynamically updated on ``localhost:3000``.

Since Flask is already started in debug mode, you can do your development/update/changes for server side (Flask) in
`${FEDBIOMED_DIR}/gui/server`. React part (ui) on development mode will call API endpoint from `localhost:8484`, this is why
first you should start Flask server first.

After development/debugging is done. To update changes in built GUI, you need to start GUI with ``--recreate`` command. Afterward,
you will be able to see changes on the ``localhost:8484`` URL which serve built UI files.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run data-folder ../data gui --recreate start
```
