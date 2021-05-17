# Fed-BioMed network

version 3 implementation of the Fed-BioMed project - network infrastructure component

## development environemnt

### conda environments for development

* to setup or update the environments, you may use the configure_conda.sh script:

```
$ ./scripts/configure_conda
```

* the **fedbiomed** environment contains all common packages for the other environements
* there is one specific environemnt for each component:

  * fedbiomed-network :  provides HTTP upload/download server and MQQT daemon
  * fedbiomed-researcher : provides the researcher part
  * fedbiomed-node : provides the client part

* conda environement are cumulative, meaning that you must activate **fedbiomed** before a specific environement. eg:

```
$ conda activate fedbiomed
$ conda activate fedbiomed-researcher
```

### run the network part

* run the following script in a terminal:

```
$ source ./scripts/fedbiomed_environment network
```

* the script will setup up all environment variables to start the network part

* deploy the docker containers

```
$ cd envs/development/network
$ ./deploy.sh --local
```
