---
title: Installation
description: Fed-BioMed installation
keywords: Fed-BioMed, Installation, Federated Learning
---

# Fed-BioMed software installation

This tutorial gives steps for installing Fed-BioMed components (node, researcher).

!!! note "Deployment"
    [Deployment documentation](../../user-guide/deployment/deployment.md) explains other available setups.


## System requirements

Fed-BioMed is developed and tested under  Linux and MacOS distributions. It has not been fully tested on Windows machines. Using Fed-BioMed on windows can raise unexpected error.


## Software packages

Fed-BioMed is a Python-based framework that can be installed using the `pip` package manager. Therefore, please ensure that Python is installed on your system.

Fed-BioMed does not support a wide range of Python versions. Currently, the required Python version is **3.10**. To configure a specific Python version in your workspace, it is recommended to use tools such as [Conda](https://docs.conda.io/) or [pyenv](https://github.com/pyenv/pyenv).


!!! info "Docker"
    Docker is only needed for [advanced usage scenarios](../../user-guide/deployment/deployment.md) with additional VPN protection of Fed-BioMed communications.


## Install Fed-BioMed

The command below will perform a complete installation of Fed-BioMed. This installation allows you to test all Fed-BioMed functionalities.

```
pip install "fedbiomed[node, researcher] @ git+https://github.com/fedbiomed/fedbiomed.git"
```

Fed-BioMed consists of different components, each requiring specific dependencies. These components are `node` and `researcher`. In the context of Federated Learning, these components are typically used in different locations and environments. To avoid installing unnecessary packages that may not be used, the dependencies for these components have been made optional in the pip package.

If you only need to install the `node` or the `researcher` component, you can use one of the following commands:

For `node` only installation:
```
pip install "fedbiomed[node] @ git+https://github.com/fedbiomed/fedbiomed.git"
```

For `researcher` only installation:
```
pip install "fedbiomed[researcher] @ git+https://github.com/fedbiomed/fedbiomed.git"
```

For installing optional node GUI:
```
pip install "fedbiomed[gui] @ git+https://github.com/fedbiomed/fedbiomed.git"
```

Fed-BioMed is provided under [Apache 2.0 License](https://github.com/fedbiomed/fedbiomed/blob/master/LICENSE.md).



## The Next Step

After the steps above are completed you will be ready to start Fed-BioMed components. In the following tutorial you will learn how to launch components and add data in Fed-BioMed to prepare an experiment.
