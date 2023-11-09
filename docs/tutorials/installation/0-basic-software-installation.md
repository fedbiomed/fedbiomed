---
title: Installation
description: Installation tutorial for Fed-BioMed
keywords: Fed-BioMed, Installation, Federated Learning
---

# Fed-BioMed software installation

This tutorial gives steps for installing Fed-BioMed components (network, node, researcher) on a single machine.
[Deployment documentation](../../user-guide/deployment/deployment.md) explains other available setups.

<iframe width="560" height="315" src="https://www.youtube.com/embed/X4TSDdIqeLM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Hardware requirements

* 8GB RAM minimum for handling PyTorch & ML packages

## System requirements

Fed-BioMed is developed and tested under up to date version of :

* **Linux Ubuntu and Fedora**, should also work or be easily ported under most Linux distributions
* **MacOS**

Check specific guidelines for installation on [Windows 11](../../user-guide/installation/windows-installation.md).


## Software packages

 The following packages are required for Fed-BioMed :

 * [`docker`](https://docs.docker.com)
 * [`docker compose` v2](https://docs.docker.com/compose): don't confuse it with the obsolete `docker-compose` v1
 * [`conda`](https://conda.io)
 * `git`


### Install `docker` and `docker compose`

#### Linux Fedora

Install and start [docker engine packages](https://docs.docker.com/engine/install/fedora/). In simple cases it is enough to run :

```
$ sudo dnf install -y dnf-plugins-core
$ sudo dnf config-manager \
    --add-repo \
    https://download.docker.com/linux/fedora/docker-ce.repo
$ sudo dnf install -y docker-ce docker-ce-cli containerd.io
$ sudo systemctl start docker
```

Allow current account to use docker :

```
$ sudo usermod -aG docker $USER
```

Check with the account used to run Fed-BioMed that docker is up and can be used by the current account without error :

```
$ docker run hello-world
```

Install `docker compose` and `git` :
```
$ sudo dnf install -y docker-compose-plugin git
```

#### MacOS

Install `docker` and `docker compose` choosing one of the available options for example :

* official full [Docker Desktop](https://docs.docker.com/desktop/mac/install/) installation process, please check product license
* your favorite third party package manager for example :
    * macports provides [docker](https://ports.macports.org/port/docker/) and [git](https://ports.macports.org/port/git/) ports
    * homebrew provides [docker](https://formulae.brew.sh/formula/docker) and [git](https://formulae.brew.sh/formula/git) formulae
    * don't use the `docker-compose` v1 from macports or homebrew !
    * for `docker compose` v2, adapt the [manual plugin install procedure](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually) by picking the [proper binary for your hardware](https://github.com/docker/compose/releases)

Check with the account used to run Fed-BioMed docker is up and can be used by the current account without error :

```
$ docker run hello-world
```

#### Other

Connect under an account with administrator privileges, install [`docker`](https://docs.docker.com/engine/install), ensure it is started and give docker privilege for the account used for running Fed-BioMed. Also install [`docker compose` v2](https://docs.docker.com/compose/install/) and `git`

Check with the account used to run Fed-BioMed docker is up and can be used by the current account without error :

```
$ docker run hello-world
```


### Install conda

#### Linux Fedora

Simply install the package :

```
$ sudo dnf install conda
```

Check conda is properly initialized with the following command that should answer the default `(base)` environment:

```
$ conda env list
```

#### Other

Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) package manager.

During the installation process, let the conda installer initialize conda (answer "yes" to “Do you wish the installer to initialize Anaconda3 by running conda init ?”)

Check conda is properly initialized with the following command that should answer the default `(base)` environment:
```
$ conda env list
```

## Fed-BioMed software
<div id="install-fedbiomed-software" class="anchor">
</div>

Download Fed-BioMed software by cloning the git repository :

```
$ git clone -b master https://github.com/fedbiomed/fedbiomed.git
$ cd fedbiomed
```

In the following tutorials, Fed-BioMed commands use a path relative to the base directory of the clone, noted as `${FEDBIOMED_DIR}`. This is not required for Fed-BioMed to work but enables you to run the tutorials more easily.

The way to setup this directory depends on your operating system (Linux, macOSX, Windows), and on your SHELL.

As an example, on Linux/macOSX with bash/zsh it could be done as:

```
export FEDBIOMED_DIR=$(pwd)
```

or

```
export FEDBIOMED_DIR=${HOME}/where/is/fedbiomed
```

Remember, that this environment variable must be initialized (to the same value) for all running shells (you may want to declare it in your shell initialization file).

Fed-BioMed is provided under [Apache 2.0 License](https://github.com/fedbiomed/fedbiomed/blob/master/LICENSE.md).

We don't provide yet a packaged version of Fed-BioMed (conda, pip).


## Conda environments

Fed-BioMed uses conda environments for managing package dependencies.

Create or update the conda environments with :

```
$ ${FEDBIOMED_DIR}/scripts/configure_conda
```

List the existing conda environments and check the 3 environments `fedbiomed-gui` `fedbiomed-node` `fedbiomed-researcher` were created :

```
$ conda env list
[...]
fedbiomed-network        /home/mylogin/.conda/envs/fedbiomed-gui
fedbiomed-node           /home/mylogin/.conda/envs/fedbiomed-node
fedbiomed-researcher     /home/mylogin/.conda/envs/fedbiomed-researcher
[...]
```

!!! note "Conda environment for Fed-BioMed Network Component"
    The Fed-BioMed network component, which consists of two modules, namely `MQTT` and `Restful`, runs in a Docker container. Although `fedbiomed-network` exists as a conda environment, it is not directly used on the local system where Fed-BioMed is installed. However, it can be used to launch the `restful` module locally instead of using Docker.
    
    The conda environment can be installed with the following command

    ```
    ${FEDBIOMED_DIR}/scripts/configure_conda network
    ```

!!! note "Conda environment for Fed-BioMed Node GUI"
    Fed-BioMed comes with a user interface that allows data owners (node users) to deploy datasets and manage requested 
    training plans easily. The `fedbiomed-gui` environment is not going to be used unless the Node GUI is launched. 
    If you don't plan on using the GUI, you can install only the `fedbiomed-node` and `fedbiomed-researcher` environments.

    ```
    ${FEDBIOMED_DIR}/scripts/configure_conda node researcher
    ```
    
    Please see [Node GUI user guide](../../user-guide/nodes/node-gui.md) to get more information about launching GUI on your local machine.



## The Next Step

After the steps above are completed you will be ready to start Fed-BioMed components. In the following tutorial you will learn how to launch components and add data in Fed-BioMed to prepare an experiment.
