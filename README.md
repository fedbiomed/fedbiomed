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

* to create or update the environments, you can use the **configure_conda** script:

```
$ ./scripts/configure_conda
```

* this script will create the environments, or update them if the conda.yaml files have been modified

* there is one specific environment for each component:

  * fedbiomed-network    : provides environement for HTTP upload/download server and MQQT daemon
  * fedbiomed-researcher : provides environement for the researcher part
  * fedbiomed-node       : provides environement for the node part

**Remark**:

On macOSX, you may encounter conflicts with the default provided conda YAML files.
To resolve this issue, we provide a set of (_yet experimental_) alternative YAML files.
In order to install them, you can use the **configure_conda** script with the **-x** flag.

```
$ conda activate base
$ ./scripts/configure_conda -x
```

**Warning**:

* mixing default environment and experimental environment may be tricky
* in case of errors, you remove all environements ans restart from fresh.


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


## Using Tensorboard

To enable tensorboard during traning routine to see loss values, you need to set `tensorboard` parameter to `true` while initializing Experiment class.

```
exp = Experiment(tags=tags,
                 #nodes=None,
                 model_path=model_file,
                 model_args=model_args,
                 model_class='MyTrainingPlan',
                 training_args=training_args,
                 rounds=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None,
                 tensorboard=True
                )
```

During training, the scalar values (loss) will be writen in the `runs` directory. You can either start tensorboard from jupyter notebook or terminal window.

**Start tensforboard from notebook**

First you should import ROOT_DIR from researcher environment in another cell

`from fedbiomed.researcher.environ import ROOT_DIR`

Load tensroboard extension in a different code block.

`%load_ext tensorboard`

Run following command to start tensorboard

`tensorboard --logdir "$ROOT_DIR"/runs`

**Start tensorboard from terminal windows**

- Open new terminal and cd into fedbiomed root directory and run following command.

**Note:** Please make sure that already activated fedbiomed-researcher conda environment.
.

`tensorboard --logdir $PYTHONPATH/runs`


## Model Hashing and Enabling Model Approve

Fed-BioMed offers optional model approval feature to approve the models requested by the researcher. This model approval process is done by hashing/checksum oparation by the ModelManager of node instance. When the `MODEL_APPROVE` mode is enabled, node should register/approve model files before performing the training. For testing and easy development, there are already presented default models by Fed-BioMed for the tutorials that we provide in the `notebooks` directory. However, node can also enable or disable the mode for allowing default models to perform training.

#### Config file for security parameters

Enabling model approval mode, allowing default Fed-BioMed models and the hashing algorithm that will be performed for the checksum oparation can be configurred from the config file of the node. The following code snippet represents an example security section of config file with default values.

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

By default, when node is started/add-data for the first time without additional security parameters, `model_approval` mode comes as disable. If `model_approval` is disabled the status of `allow_defaults_models` will have no effect. To enable `model_approval` you should set `model_approval` to `True` and if it is desired `allow_default_models` can be set to `False` to not accepting models of default Fed-BioMed examples.

The default hashing algorithm is `SHA256` and it can also be changed to other hashing algorithms that are provided by Fed-BioMed. You can see the list of Hashing algorithms in the following section.


#### Hashing Algorithms

`ModelManager` provides different hashing algorithms, and the algorithm can be changed through the config file of the node. The name of the algorithms should typed with capital letters. However, after changing hashing algorithm node should be restarted because it checks/updates hashing algorithms of the register/default models during the starting process.

Provided hashing algorithms are `SHA256`, `SHA384`, `SHA512`, `SHA3_256`, `SHA3_384`, `SHA3_512`, `BLAKE2B` and `BLAKE2S`. These are the algorithms that has been guaranteed by `hashlib` library of Python.


#### Starting nodes with different modes

To enable `model_approval` mode and `allow_default_models` node can be started following command.

```shell
./scripts/fedbiomed_run node config config-n1.ini --enable-model-approval --allow-default-models start
```

This command will start the node with in model approval mode even the config file has been set as `model_aprove = False`.However it doesn't change the config file. If there is no config file named `config-n1.ini` it creates a config file for the node with enabled model approved mode.

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

Default models has been located at the `env/development/default_models/` directory as `txt` files. Each time when the node started with the `model_approval = True` and `allow_default_model = True` modes, hashing of the model files are get checked to detect if the file is modified, the hashing algorithm has changed or is there any new model file added. If model files are modified `ModelManager` updates hashes for these models in the database. If the hashing algoritmh of the model is different that the active hashing algorithm, hashes also get updated. This process only occurs when both `model-approval` and `allow-default-models` modes are activated. To add new default model for the examples or for testing, model files should be saved as `txt` and copied into the `envs/development/default_models` directory. After the copy/save oparation node should be restarted.


#### Registering New Models

New models can be registered using `fedbiomed_run` scripts with `register-model` option.

```shell
./scripts/fedbiomed_run node config config-n1.ini register-model
```

The CLI will ask for name of the model, description and the path where model file is stored. **Model files should saved as txt in the file system for registiration.** This because these files are for only hashing purposes not for loading modules.

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

Default models can not be removed using fedbiomed CLI. They should be removed from the `envs/development/default_models` directory. After restarting the node, deleted model files will be also removed from the `Models` table of the node DB.


#### Updating Registered model

Following command is used for updating registered models. It updates chosen model with provided new model file. User also
can provide same model file to update its content.

```
./scripts/fedbiomed_run node config config-n1.ini update-model
```
