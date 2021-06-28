# Fed-BioMed network

version 3 implementation of the Fed-BioMed project - network infrastructure component

## development environment

### setup conda environments

* to create or update the environments, you use the **configure_conda** script:

```
$ ./scripts/configure_conda
```

* this script will create the environments, or update them if the conda.yaml files have been modified

* there is one specific environment for each component:

  * fedbiomed-network :  provides HTTP upload/download server and MQQT daemon
  * fedbiomed-researcher : provides the researcher part
  * fedbiomed-node : provides the client part

### activate the environments

In a terminal, you can configure environments to work interactively inside a specific repository, with the right conda environment and the right PYTHONPATH environment.

**WARNING**: this script only work for **bash** and **zsh**. It is not compliant with ksh/csh/tcsh/etcsh/...

```
source ./scripts/fedbiomed_environment ENV
```

where ENV chosen from:

* network (work inside fedbiomed-network)
* node (work inside fedbiomed-node)
* researcher (work inside fedbiomed-researcher)


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
$ ./scripts/fedbiomed_run node start another_config.ini
```


#### run a researcher notebook

* in a new terminal:

```
$ ./scripts/fedbiomed_run researcher
```

* this will launch a new jupyter notebook


#### run a researcher script

* in a new terminal:

```
$ source ./scripts/fedbiomed_environment researcher
```

* then you can use any researcher script

```
$ python ../fedbiomed-researcher/notebooks/getting-started-localhost.py
```

### clean state (restore environments back to new)

De-configure environments, remove all configuration files and caches
```
source ./scripts/fedbiomed_environment clean
```



## some tools to help debugging

### lqueue

list the content of a message queue (as used in fedbiomed.node and fedbiomed.researcher)

usage:  lqueue directory
   or
        lqueue dir1 dir2 dir3 ...
