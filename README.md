# Fed-BioMed network

version 3 implementation of the Fed-BioMed project - network infrastructure component

## development environemnt

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
