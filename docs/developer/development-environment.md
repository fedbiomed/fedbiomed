# Configuring Development Environment

This article is written to guide developers to create a development environment. This guide can also be used to build Fed-BioMed software from its source.

This article will guide you to:

- Clone Fed-BioMed source repository.
- Setup a recommended tool to manage different Python versions.
- Setup some optional tools to manage isolated python environments such as `conda` and `virtualenv`.
- Setup pdm to manage dependencies and build operations, and it usage.


## Clone the repository
The first step is to clone the repository please go to [Fed-BioMed GitHub Repository](https://github.com/fedbiomed/fedbiomed) and clone it your local environment.

```
git clone git@github.com:fedbiomed/fedbiomed.git <path-to-clone>
cd <path-to-clone>
```


## Compatible Python version

Before you start using Fed-BioMed or developing on Fed-BioMed please make sure that you have the required Python version is installed on your machine. 
Please go to `pyproject.toml` file located in the root of Fed-BioMed clone, and check the 
required Python version interval.

!!! note "Tool for different Python version"

    You can use the tool `pyenv` to be able to install specific Python versions into your local environment
    ```
    pyenv install 3.11
    pyenv global 3.11
    ```

    The code above will install latest Python 3.11, and activate this version globally for your machine.


## Install recommended tools to manage development environment

### Use a virtual environments to manage dependencies

Using virtual environments will allow you to isolate your Fed-BioMed development environment from your other projects. There, you have of course several options. You can use built-in python `venv` or `conda` or any other compatible Python virtual environment tools.


While `conda` also allows installing a specified Python version, `venv` requires that Python and Pip are already installed. Therefore, if `conda` is used, additional tools like `pyenv` are not necessary for managing Python versions. However, if `venv` is preferred, using tools like `pyenv` to install the required Python version is recommended.


#### Case: venv

The advantage of `venv` is that it keeps all project dependencies within your project folder and is part of `python` standard library. Please go to Fed-BioMed project root (Fed-BioMed) clone and execute following command.

```
cd <path-to-fedbiomed-clone>
python -m venv ./dev-env
```

The command above will generate `dev-env` folder in the project folder where all the project dependencies are kept.

To activate the environment you can execute the following command from project root directory.

```
source dev-env/bin/activate
```

You can also use default naming convention `.venv` for virtual environment name.

#### Case: Conda

`conda` can install Python version directly in the environment that is created. The command below will create a virtual environment and install the specified Python version.

```
conda create --name fedbiomed-dev python=3.14
conda activate fedbiomed-dev
```


### Install Pdm

Once virtual environments are set `PDM` package manager can be installed via `PyPI`. It is also possible to install `PDM` globally for the Python version that is activated or base `conda` environment if `conda` is preferred as virtual environment manager.

```
pip install pdm
```

The command above will install PDM in the active `conda` environment or in active `pyenv` if it is used for managing Python versions.


### NodeJS and yarn

Node.js and Yarn are required for frontend development and for building the
GUI distribution from source. The current GUI build hook runs `yarn install` and `yarn build`, including during editable installation unless explicitly skipped using `FBM_SKIP_FRONTEND_BUILD=1`.
Core and API builds do not require frontend tools. 

!!! warning "Important" 
    Installing a released GUI wheel uses its bundled frontend assets and does not require Yarn, unless the user explicitly commands to rebuild the GUI. If you wish to rebuild the GUI for frontend development, install Yarn and NodeJS through their official documentations [Node.js](https://nodejs.org/en/download) and [Yarn Classic](https://classic.yarnpkg.com/en/)

## Using PDM

### Projects, workspace, and dependency groups

For developers, Fedbiomed supports the new experimental feature of pdm for configuring multiple packages using a single lock file. If you want to use this feature, make sure that you have the PDM version 2.28 or above. You can check it with `pdm --version`; Fedbiomed currently does not pin a PDM version as a dependency. Development uses the root `pdm.lock` file in the project root as the single lock file. The package dependency chain is `fedbiomed-gui` → `fedbiomed-node-api` → `fedbiomed`.

If you do not wish to use this experimental feature, with your preferred environment manager, you can just use the 3 separate `pyproject.toml` files to install the dependencies. The `.toml` files are located in the root of each package:  root `pyproject.toml` for `fedbiomed`, `fedbiomed_node_api/pyproject.toml` for the API, and `fedbiomed_gui/pyproject.toml` for the GUI. 

You can use `pdm sync -G local` to install `fedbiomed`, `fedbiomed-node-api` and `fedbiomed-gui` packages in one go. The `local` group is provided for easy installation for developers, it installs all three packages in editable mode. Make sure you have selected the correct environme t before with `pdm use --venv MYENV`, if you are using pdm to manage your virtual environments. Details on using pdm to manage venvs can be found in [PDM environment selection](https://pdm-project.org/en/latest/usage/venv/).

The other pdm groups and what they install can be seen below:

| Group | Kind | Purpose |
| --- | --- | --- |
| `node` | Published extra | Compatibility extra; currently empty because core includes the basic node. |
| `researcher` | Published extra | Researcher tools, including Jupyter, TensorBoard, and matplotlib. |
| `node-api` | Published extra | Matching `fedbiomed-node-api` distribution and its web dependencies. |
| `gui` | Published extra | Matching GUI distribution, which depends on the API. |
| `local` | Development group | Both API and GUI packages installed editable from this checkout. |
| `test` | Development group | pytest, tox, coverage, and packaging test tools. |
| `dev` | Development group | Ruff and pre-commit. |
| `docs` | Development group | MkDocs and documentation plugins. |

Published extras are defined in `[project.optional-dependencies]`; development
groups are not extras available through `pip install "fedbiomed[...]"`.

### Verify a fresh installation

After using pdm sync, you can verify the distributions are installed correctly with the command below:

```sh
pdm run python -c 'from importlib.metadata import version; print({n: version(n) for n in ("fedbiomed", "fedbiomed-node-api", "fedbiomed-gui")})'
```

### What about instant changes in source code?

PDM automatically adds package source to your environment that makes `fedbiomed` module accessible from Python interpreter. It also handles instant changes in the source code. If a new module is added or a module is edited PDM will automatically make changes available within the environment currently used. It means there is no need to install or rebuild `fedbiomed` package.

!!! warning "Important"
    You should never install `fedbiomed` from `pip` in your development environment. This may interrupt to access current source code changes in your development environment.


### Add new module

New packages can be added through `pyproject.toml`, or using the command `pdm add`. Please see `pdm add --help` for detailed usage instructions.

Once a new package is added via `pyproject.toml` it may be required to run `pdm lock -G :all` to resolve the dependencies, then, execute `pdm install` to update packages and install missing ones.


### For more

Please visit `pdm` documentation for more information and usage details.


## Development/Debugging for GUI

If you want to customize or work on user interface for debugging purposes, it is always better to use ReactJS in development mode, otherwise building GUI
after every update will take a lot of time. To launch user interface in development mode first you need to start Flask server. This can be
easily done with the previous start command. Currently, Flask server always get started on development mode.  To enable debug mode you should add `--debug`
flag to the start command.

```shell
# data folder defaults to `/path/to/my-node/data`
fedbiomed node -p /path/to/my-node gui start --debug

# Or use an alternate data path 
# fedbiomed node -p /path/to/my-node gui start --data-folder /alternate/data-path --debug
```

**Important:** Please do not change Flask port and host while starting it for development purposes. Because React (UI) will be calling
``localhost:8484/api`` endpoint in development mode.

The command above will serve the web application and the API services. It means that on the URL `localhost:8484` you will be able to see the user interface. This user interface won't be updated automatically because it is already built. To have dynamic update for user interface you can start React with ``yarn start``.

```shell
# use the Python environment selected above
cd ${FEDBIOMED_DIR}/packages/node-ui/ui
yarn start
```

After that if you go ``localhost:3000`` you will see same user interface is up and running for development.  When you change the source codes
in ``${FEDBIOMED_DIR}/fedbiomed_gui/ui/src`` it will get dynamically updated on ``localhost:3000``.

Since Flask is already started in debug mode, you can do your development/update/changes for server side (Flask) in `${FEDBIOMED_DIR}/fedbiomed_gui/server`. React part (ui) on development mode will call API endpoint from `localhost:8484`, this is why first you should start Flask server first.

After development/debugging is done, rebuild the React app to see the changes on the ``localhost:8484`` URL, which serves
the built UI files. `--recreate` rebuilds it before starting the server:

```shell
fedbiomed node gui start --recreate
```

## Building the distributions

If you wish to build your own Python wheels for publidcation: Core, Node API, and GUI have separate build configurations. From the repository root, you can build their wheels and source archives with:

```sh
pdm build
pdm build -p fedbiomed_node_api
pdm build -p fedbiomed_gui
```

Core includes `fedbiomed` and its existing shared resources; the API includes
`fedbiomed_node_api` and its configuration template; the GUI includes
`fedbiomed_gui` and compiled frontend assets. The GUI build will require Yarn.
Its source archive also contains the frontend sources and build hook.

To reuse a prebuilt `fedbiomed_gui/ui/build` directory, run
`FBM_SKIP_FRONTEND_BUILD=1 pdm build -p fedbiomed_gui`.

To check package contents and to test rebuilding the wheels from the created source archives:

```sh
python -m pytest --noconftest -c /dev/null tests/test_package_builds.py -q
```

This packaging check runs without application dependencies or frontend tools.

## Troubleshooting

You may encounter some common issues during installation or after the installation due to some missing packages. Please visit [troubleshooting](../support/troubleshooting.md) dedicated page for common issues.


### Error on MacOS/Ubuntu regarding `pyenv` usage:

- **`_lzma` module not found**
    - If you are using **MacOS** and installing Python versions through `pyenv` you may have some missing packages in your environment. `ModuleNotFoundError: No module named '_lzma'` is one of them. If you faced this error please install `brew install xz`, and reinstall python version `pyenv install <version>`.


    - If you are using **Ubuntu** or **Debian based OS**, and install python through `pyenv`, and get `ModuleNotFoundError: No module named '_lzma'`
        This could be fixed by using
        ```bash
        sudo apt install liblzma-dev
        ```
- **Other issues**
    - For other issues, `pyenv` comes with an utility [`pyenv-doctor`](https://github.com/pyenv/pyenv-doctor), made for checking `pyenv` installation and dependencies, and can be run with: `pyenv doctor`. It will try to find the issue with your installation and proposes appropriate solutions to your issue.

    More troubleshooting for `pyenv` can be found [here](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).


### Building Fed-BioMed Takes too Long

Some static files located in the root Fed-BioMed source directory (e.g., notebooks, tests, etc.) are also included in the final distribution. Therefore, having large data files or artifacts left from operations for testing and development purposes can increase the loading time. Please ensure that such data files are cleared before building the Fed-BioMed package to reduce build time.
