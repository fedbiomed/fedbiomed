# Fed-BioMed

version 3 implementation of the Fed-BioMed project

## Install and run in development environment

Fed-BioMed is developped under Linux Fedora, should be easily ported to other Linux distributions.

### clone repos

Clone the repositories for the Fed-BioMed components under the same base directory :
```
git clone git@gitlab.inria.fr:fedbiomed/fedbiomed-network.git # this one
git clone git@gitlab.inria.fr:fedbiomed/fedbiomed-node.git
git clone git@gitlab.inria.fr:fedbiomed/fedbiomed-researcher.git
```

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
$ ./scripts/fedbiomed_run node start config another_config.ini
```


#### run a researcher notebook

* in a new terminal:

```
$ ./scripts/fedbiomed_run researcher
```

* this will launch a new jupyter notebook
  - `getting-started.ipynb` : training a simplenet + federated average on MNIST data
  - `local_training.ipynb` : comparing the simplenet + federated average on MNIST data with its local training equivalent


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



## Misc developper tools to help debugging

### lqueue

list the content of a message queue (as used in fedbiomed.node and fedbiomed.researcher)

usage:  lqueue directory
   or
        lqueue dir1 dir2 dir3 ...

## Developper info on continuous integration

Continuous integration uses a [Jenkins](https://www.jenkins.io/) server on `ci.inria.fr`. 

CI tests are triggered automatically by gitlab on a :
* merge request to `develop` or `master` branch
* push in `develop`, `master`, `feature/test_ci` branches (eg: after a merge, pushing a fix directly to this branch)

The merge should not be completed before CI pipeline succeeds
* pushing a fix to the branch with the open merge request re-triggers the CI test
* CI test can also be manually triggered by adding a comment to the merge request with the text `Jenkins please retry a build`

CI pipeline currently contains :
* running a simplenet + federated average training, on a few batches of a MNIST dataset, with 1 node. For that, CI launches `./scripts/CI_build` (wrapping for running on CI server) which itself calls `./scripts/run_test_mnist` (payload, can also be launched on localhost)
  - clone the Fed-BioMed repositories, set up condas and environment, launch network and node. 
  - choose an existing git branch for running the test for each of the repos, by decreasing preference order : source branch of the merge, target branch of the merge, `develop`
  - launch the `fedbiomed-researcher` script `./scripts/getting-started-localhost.py`
  - test succeeds if the script completes without failure.


To view CI test output and logs :
* view the merge request in gitlab (select `Merge requests` in left bar, then select your merge request)
* click on the `Pipeline` number (eg: #1289345) in the merge request, then click on the `Jobs` tab, then click on the job number (eg: #1294521)
* connect with your account on `ci.inria.fr`. To get an account on `ci.inria.fr` you need to be approved by one member of the Fed-BioMed CI project or to be a member of Inria
* select `Console output` in the left pane

To configure CI test :
* connect with your account on `ci.inria.fr`
* request the Fed-BioMed team to become a member of the Fed-BioMed CI project

Note: using branch `feature/test_ci` can be useful when testing/debugging the CI setup (triggers CI on every push, not only on merge request).
