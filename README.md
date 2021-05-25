# Fed-BioMed network

version 3 implementation of the Fed-BioMed project - network infrastructure component

## development environment

### conda environments for development

* to setup or update the environments, you use the **configure_conda** script:

```
$ ./scripts/configure_conda
```

* this script will setup the environments, or update them if the conda.yaml files have been modified

* there is one specific environemnt for each component:

  * fedbiomed-network :  provides HTTP upload/download server and MQQT daemon
  * fedbiomed-researcher : provides the researcher part
  * fedbiomed-node : provides the client part

### tool to setup the environment

In a terminal, you can setup environments to work interactively inside a specific repository, with the right conda environment and the right PYTHONPATH environement.

```
./scripts/fedbiomed_environment ENV
```

where ENV chosen between:

* network (work inside fedbiomed-network)
* node (work inside fedbiomed-node)
* researcher (work inside fedbiomed-researcher)



### run the network part


* in a new terminal:

```
$ ./scripts/fedbiomed_run network
```

* this will start the necessary docker container (file repository and mqtt)

### run the node part

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


### run a researcher notebook

* in a new terminal:

```
$ ./scripts/fedbiomed_run researcher
```

* this will launch a new jupyter notebook
