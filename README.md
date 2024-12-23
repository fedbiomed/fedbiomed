[![Documentation](https://img.shields.io/badge/Documentation-green)](https://fedbiomed.org)
[![](https://img.shields.io/badge/Medium-black?logo=medium)](https://medium.com/fed-biomed)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/fedbiomed/fedbiomed/blob/master/LICENSE.md)
[![Python-versions](https://img.shields.io/badge/python-3.10-brightgreen)](https://www.python.org/)
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

### prerequisites

* `git`
* `python` compatible version (currently 3.10)

A recommended practice is to use install `python` in a local environment for example using `pyenv`

```
pyenv install 3.10
mkdir clone_dir ; cd clone_dir
pyenv local 3.10
```

A recommended practice is to use a virtual environment for managing dependencies. 
For example, if using `venv`:

```
python -m venv fb_env
source fb_env/bin/activate
```


### quick install

Clone the Fed-BioMed repository for running the software :

```
git clone -b master https://github.com/fedbiomed/fedbiomed.git
cd fedbiomed
```

```
pdm install
```

This is later refered as "the environment where Fed-BioMed is installed".

More details in the [developer environment installation documentation](https://fedbiomed.org/latest/developer/development-environment.md/)


### run the software

#### run the node part

* in a new terminal:

```
$ fedbiomed node start
```

* this will launch a new node

* you may also upload new data on this node with:

```
$ fedbiomed node dataset add
```

* you may also specify a new config file for the node (usefull then running multiple test nodes on the same host)

```
$ fedbiomed node --path ./my-second-node start
```

* if you want to change the default IP address used to join the fedbiomed researcher component (localhost), you can provide it at launch time:

```
$ FBM_RESEARCHER_IP=192.168.0.100 fedbiomed node start
$ FBM_SERVER_HOST=192.168.0.100 fedbiomed researcher start
```

(adjust the 192.168.0.100 IP address to your configuration)

If this option is given at the first launch or after a clean, it is saved in the configuration file and becomes the default for subsequent launches. If this option is given at a subsequent launch, it only affects this launch.

#### run a researcher notebook

* in a new terminal:

```
$ fedbiomed researcher start
```

* this will launch a new jupyter notebook working in the **notebooks** repository. First try:

  - `101_getting-started.ipynb` : training a simplenet + federated average on MNIST data


#### run a researcher script

1. in a new terminal: activate the environment where Fed-BioMed is installed
2. convert the notebook to a python script
```bash
jupyter nbconvert --output=101_getting-started --to script ./notebooks/101_getting-started.ipynb
```
3. then you can use any researcher script

```bash
$ python ./notebooks/101_getting-started.py
```

### clean state (restore environments back to new)

To clean your Fed-BioMed instance :

* stop the researcher : shutdown the notebook kernel (`Quit` in on the notebook interface or `ctrl-C` on the console)
* stop the nodes : interrupt (`ctrl-C`) on the nodes console
* remove all configuration files, dataset sharing configuration, temporary files, caches for all Fed-BioMed components with :

```
$ rm -rf COMPONENT_DIR
```

Where `COMPONENT_DIR` is:
* for a node, the parameter provided as `fedbiomed node -p COMPONENT_DIR` or by default `fbm-node` if no parameter was given
* for a researcher, the parameter provided as `fedbiomed researcher -p COMPONENT_DIR` or by default `fbm-researcher` if no parameter was given


## Fed-BioMed Node GUI

Node GUI provides an interface for Node to manage datasets and deploy new ones. GUI consists of two components, Server and UI. Server is developed on Flask framework and UI is developed using ReactJS. Flask provides API
services that use Fed-BioMed's DataManager for deploying and managing dataset. All the source files for GUI has been
located on the `${FEDBIOMED_DIR}/gui` directory.

### Starting GUI

Node GUI can be started using Fed-BioMed CLI.

```shell
fedbiomed node [--path [COMPONENT_DIRECTORY]] gui --data-folder '<path-for-data-folder>' start
```

Arguments:

- ``data-folder``: Data folder represents the folder path where datasets have been stored. It can be absolute or relative path. If it is relative path, Fed-BioMed base directory is going to be used as reference. **If `datafolder` is not provided. Script will look for
`data` folder in the Fed-BioMed root directory and if it doesn't exist it will raise an error.**
- ``--path``: Component directory  whose GUI will be launched which is going to be used for GUI. If it is not
provided, default will be `fbm-node` that is created in directory where the command is executed. If component directory is not existing a default node component will instantiated in the given directory if the parent directory is existing.

It is also possible to start GUI on specific host and port, By default it is started `localhost` as host and `8484` as port.  To change
it you can modify following command.

The GUI is based on HTTPS and by default, it will generate a self-signed certificate for you. Butyou can also start GUI specifying the certificate and the private key
names you want to use for HTTPS support. **Please note that they must be in `${FEDBIOMED_DIR}/etc` folder.**

```shell
fedbiomed node --path <path/to/component/directory> gui --data-folder '<path-for-data-folder>' ' cert '<name-of-certificate>' key '<name-of-private-key>' start
```

**IMPORTANT:** Please always consider providing `data-folder` argument while starting the GUI.

```shell
fedbiomed node -d my-node gui data-folder ../data  --port 80 --host 0.0.0.0 start

```

### Details of Start Process

When the Node GUI is started, it installs `npm` modules and builds ReactJS application in ``${FEDBIOMED_DIR}/var/gui-build``. If the GUI
is already built (means that `gui/ui/node_modules` and `var/gui-build` folders exist), it does not reinstall and rebuild ReactJS. If you want to
reinstall and rebuild, please add `--recreate` flag in the command same as below,

```shell
fedbiomed node gui data-folder ../data --recreate start
```


### Launching Multiple Node GUI

It is possible to start multiple Node GUIs for different nodes as long as the http ports are different. The
commands below starts three Node GUI for the nodes; config-n1.ini, config-n2.ini and config-n3.ini on the ports respectively, `8181`, `8282` and `8383`.

```shell
fedbiomed node -d my-node gui --data-folder ../data --port 8181 start
fedbiomed node -d my-second-node gui --data-folder ../data --port 8282 start
fedbiomed node -d my-second-node gui --data-folder ../data --port 8383 start
```

### Development/Debugging for GUI

If you want to customize or work on user interface for debugging purposes, it is always better to use ReactJS in development mode, otherwise building GUI
after every update will take a lot of time. To launch user interface in development mode first you need to start Flask server. This can be
easily done with the previous start command. Currently, Flask server always get started on development mode.  To enable debug mode you should add `--debug`
flag to the start command.

```shell
fedbiomed node -d my-node gui --data-folder ../data --debug start
```
**Important:** Please do not change Flask port and host while starting it for development purposes. Because React (UI) will be calling
``localhost:8484/api`` endpoint in development mode.

The command above will serve ``var/gui-build`` directory as well as API services. It means that on the URL `localhost:8484` you will be able to
see the user interface. This user interface won't be updated automatically because it is already built. To have dynamic update for user interface you can start React with ``npm start``.

```shell
# use the python environment for [development](../docs/developer/development-environment.md)
cd ${FEDBIOMED_DIR}/gui/ui
npm start
```

After that if you go ``localhost:3000`` you will see same user interface is up and running for development.  When you change the source codes
in ``${FEDBIOMED_DIR}/gui/ui/src`` it will get dynamically updated on ``localhost:3000``.

Since Flask is already started in debug mode, you can do your development/update/changes for server side (Flask) in
`${FEDBIOMED_DIR}/gui/server`. React part (ui) on development mode will call API endpoint from `localhost:8484`, this is why
first you should start Flask server first.

After development/debugging is done. To update changes in built GUI, you need to start GUI with ``--recreate`` command. Afterward,
you will be able to see changes on the ``localhost:8484`` URL which serve built UI files.

```shell
fedbiomed node gui start --data-folder ../data
```

