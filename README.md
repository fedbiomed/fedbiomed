[![Documentation](https://img.shields.io/badge/Documentation-green)](https://fedbiomed.org)
[![PyPI Downloads](https://static.pepy.tech/badge/fedbiomed)](https://pepy.tech/projects/fedbiomed)
[![](https://img.shields.io/badge/Medium-black?logo=medium)](https://medium.com/fed-biomed)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/fedbiomed/fedbiomed/blob/master/LICENSE.md)
[![Python-versions](https://img.shields.io/badge/python-3.10-brightgreen)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/fedbiomed?color=white)](https://pypi.org/project/fedbiomed/)
[![Citation](https://img.shields.io/badge/cite-paper-orange)](https://arxiv.org/abs/2304.12012)
[![PR](https://img.shields.io/badge/PRs-welcome-green)](https://github.com/fedbiomed/fedbiomed/pulls)
[![codecov](https://img.shields.io/codecov/c/gh/fedbiomed/fedbiomed/develop?logo=codecov)](https://app.codecov.io/gh/fedbiomed/fedbiomed/tree/develop)

# Fed-BioMed

## Introduction

Fed-BioMed is an open source project focused on empowering healthcare using non-centralized approaches for statistical analysis and machine learning.

The project is currently based on Python, PyTorch and Scikit-learn, and enables developing and deploying collaborative learning analysis in real-world machine learning applications, including federated learning and federated analytics.

The code is regularly released and available on the **master** branch of this repository. The documentation of the releases can be found at https://fedbiomed.org

Curious users may also be interested by the current developments, occurring in the **develop** branch (https://github.com/fedbiomed/fedbiomed/tree/develop)
Develop branch is WIP branch for next release. It may not be fully usable, tested and documented. Support is provided only for releases.


## Install and run in development environment

Fed-BioMed is developped under Linux Ubuntu & Fedora (should be easily ported to other Linux distributions) and MacOS X.
It is also ported on Windows WSL2.

This README.md file provide a quick start/installation guide for Linux.

Full installation instruction are also available at: https://fedbiomed.org/latest/tutorials/installation/0-basic-software-installation/

An installation guide is also provided for Windows11, which relies on WSL2: https://fedbiomed.org/latest/user-guide/installation/windows-installation/

### Prerequisites

- **Python:** Compatible version (currently 3.10)
- It is recommended to install Python in a local environment, for example, using `pyenv`.

```
pyenv install 3.10
pyenv local 3.10
```

A recommended practice is to use a virtual environment for managing dependencies. For example, if using `venv`:

```
python -m venv fb_env
source fb_env/bin/activate
```

### Install

Fed-BioMed can be installed using `pip` with the following command:

```bash
pip install fedbiomed[node, gui, researcher]
```

If you prefer to use Fed-BioMed in development mode, please refer to the [Developer Environment Installation Documentation](https://fedbiomed.org/latest/developer/development-environment.md).

### Quick Start: Running Fed-BioMed


#### Starting a Basic Node

To start a basic Fed-BioMed node, open a new terminal and execute the following command:

```bash
$ fedbiomed node start
```

#### Uploading New Data to the Node
To upload new data to this node, run:

```bash
$ fedbiomed node dataset add
```

#### Specifying a Component Directory
If you need to run multiple test nodes on the same host, you can specify a different component directory:

```bash
$ fedbiomed node --path ./my-second-node start
```

#### Changing the Default IP Address
To specify a different IP address for connecting to the Fed-BioMed researcher component (default: `localhost`), provide it at launch time:

```bash
$ FBM_RESEARCHER_IP=192.168.0.100 fedbiomed node start
$ FBM_SERVER_HOST=192.168.0.100 fedbiomed researcher start
```

##### Configuration Persistence
- If this option is provided at the first launch or after a clean configuration, it is saved in the configuration file and becomes the default for future launches.
- If given during a subsequent launch, it only applies to that launch without altering the saved configuration.


#### Run a Researcher Notebook

1. Open a new terminal and start the researcher component:

   ```bash
   $ fedbiomed researcher start
   ```

2. This will launch a new Jupyter Notebook environment within the **notebooks** repository. A good starting point is:

   - `101_getting-started.ipynb`: Train a SimpleNet model with federated averaging on the MNIST dataset.


#### Run a Researcher Script

1. Open a new terminal and activate the environment where Fed-BioMed is installed.

2. Convert the notebook to a Python script:

   ```bash
   jupyter nbconvert --output=101_getting-started --to script ./notebooks/101_getting-started.ipynb
   ```
3. Execute the researcher script using:

   ```bash
   $ python ./notebooks/101_getting-started.py
   ```### Clean State (restore environments back to new)

### Cleaning/Removing Fed-BioMed Components

To clean your Fed-BioMed instance:

* Stop the researcher : shutdown the notebook kernel (`Quit` in on the notebook interface or `ctrl-C` on the console)
* Stop the nodes : interrupt (`ctrl-C`) on the nodes console
* Remove all configuration files, dataset sharing configuration, temporary files, caches for all Fed-BioMed components with :

```
$ rm -rf COMPONENT_DIR
```

Where `COMPONENT_DIR` is:
* for a node, the parameter provided as `fedbiomed node -p COMPONENT_DIR` or by default `fbm-node` if no parameter was given
* for a researcher, the parameter provided as `fedbiomed researcher -p COMPONENT_DIR` or by default `fbm-researcher` if no parameter was given


## Fed-BioMed Node GUI

Node GUI provides an interface for Node to manage datasets and deploy new ones. GUI consists of two components, Server and UI. Server is developed on Flask framework and UI is developed using ReactJS. Flask provides API services that use Fed-BioMed's DataManager for deploying and managing dataset. All the source files for GUI has been located on the `${FEDBIOMED_DIR}/gui` directory.

### Starting GUI

Node GUI can be started using Fed-BioMed CLI.

```shell
fedbiomed node [--path [COMPONENT_DIRECTORY]] gui start --data-folder <path-for-data-folder>
```

Please see possible argument using `fedbiomed node gui start --help`.


It is also possible to start the GUI on a specific host and port. By default, it starts on `localhost` as the host and `8484` as the port. To change these settings, you can modify the following command. The GUI is based on HTTPS and will, by default, generate a self-signed certificate. However, you can also start the GUI by specifying the certificate and private key names you want to use for HTTPS support.

```shell
fedbiomed node --path <path/to/component/directory> gui start --data-folder <path-for-data-folder> --cert-file <path-to-certificate> --key-file <path-to-private-key>
```

```shell
fedbiomed node -p </path/to/my-node> gui --port 80 --host 0.0.0.0
```

**IMPORTANT:** Provide `data-folder` argument while starting the GUI if not using the default `/path/to/my-node/data`

```shell
fedbiomed node -p </path/to/my-node> gui --data-folder /another/data-dir --port 80 --host 0.0.0.0
```

### Launching Multiple Node GUI

It is possible to start multiple Node GUIs for different nodes as long as the http ports are different.

```shell
fedbiomed node -p my-node gui start --port 8181
fedbiomed node -p my-second-node gui start --port 8282
fedbiomed node -p my-second-node gui start --port 8383
```

Please see `docs/developer/development-environment.md` to find out how to debug and lunch UI for development purposes.

