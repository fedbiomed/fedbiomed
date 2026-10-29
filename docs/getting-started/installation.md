---
title: Installation
description: Fed-BioMed installation
keywords: Fed-BioMed, Installation, Federated Learning
---

# Fed-BioMed software installation

This tutorial gives steps for installing Fed-BioMed components (node, researcher).

!!! note "Deployment"
    [Deployment documentation](../user-guide/deployment/deployment.md) explains other available setups.

## System requirements

Fed-BioMed is developed and tested under  Linux and MacOS distributions.

It has not been fully tested on Windows under WSL2. Using Fed-BioMed on Windows can raise unexpected error.
Check specific installation guidelines for [Windows 11](../user-guide/installation/windows-installation.md).

## Hardware requirements

16GB RAM recommended, 8GB RAM minimum for handling PyTorch & ML packages, for each Fed-BioMed component.

## Software packages

Fed-BioMed is a Python-based framework that can be installed using the `pip` package manager. 
Therefore, please ensure that Python is installed on your system.

Fed-BioMed does not support a wide range of Python versions. 
Currently, the required Python versions are **3.11-3.14**. 
To configure a specific Python version in your workspace, it is recommended to use tools such as [Conda](https://docs.conda.io/) or [pyenv](https://github.com/pyenv/pyenv).

!!! info "Docker"
    Docker is only needed for [advanced usage scenarios](../user-guide/deployment/deployment.md) with additional VPN protection of Fed-BioMed communications.

## Install Fed-BioMed

The command below will perform a complete installation of Fed-BioMed. This installation allows you to test all Fed-BioMed functionalities.

```
python -m pip install "fedbiomed[node,researcher,node-api,gui]"
```

Depending on your needs, you may not need to install all Fedbiomed components(packages). Below you can select the list of packages, which you can install individually according to your needs:

| Installation | Command | Includes |
| --- | --- | --- |
| Basic node/core | `python -m pip install fedbiomed` | Core node functionality. |
| Node compatibility extra | `python -m pip install "fedbiomed[node]"` | Same as core; the extra is currently empty. |
| Researcher | `python -m pip install "fedbiomed[researcher]"` | Core and researcher tools. |
| Node API | `python -m pip install "fedbiomed[node-api]"` | Core and the separate API package. |
| Node GUI | `python -m pip install "fedbiomed[gui]"` | Core, API, and GUI packages. |

API and GUI versions are pinned to the matching core version. These commands
require the corresponding distributions to be published. Released GUI wheels
include compiled assets; installing a wheel does not require Node.js/Yarn (unless the user wishes to rebuild the GUI).
For a source checkout and unpublished packages, follow the
[development environment guide](../developer/development-environment.md#using-pdm).

For installing optional dependency to FLamby:
```
pip install git+https://github.com/owkin/FLamby@main
```

Fed-BioMed is provided under [Apache 2.0 License](https://github.com/fedbiomed/fedbiomed/blob/master/LICENSE.md).

## The Next Step

After the steps above are completed you will be ready to start Fed-BioMed components. In the following tutorial you will learn how to launch components and add data in Fed-BioMed to prepare an experiment.
