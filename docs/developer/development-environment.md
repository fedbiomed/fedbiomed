# Configuring Development Environment

This article is written to guide developers to create development environment. This guide can also be used to build Fed-BioMed software from its source.

This article will guide you to:

- Clone Fed-BioMed source repository.
- Setup a recommended tool to manage different Python versions.
- Setup some optional tools to manage isolated python environments such as `conda` and `virtualenv`.
- Setup pdm to manage dependencies and build operations, and it usage.


## Clone the repository
The first step is to clone the repository please go to (Fed-BioMed GitHub Repository)[https://github.com/fedbiomed/fedbiomed] and clone it your local environment.

```
git clone git@github.com:fedbiomed/fedbiomed.git <path-to-clone>
cd <path-to-clone>
```


## Compatible Python version

Before you start using Fed-BioMed or developing on Fed-BioMed please make sure that required Python version is installed on your machine. Please go to `pyproject.toml` file located in the root of Fed-BioMed clone, and check required Python version interval.

!!! note "Tool for different Python version"

    You can use the tool `pyenv` to be able to install specific Python versions into your local environment
    ```
    pyenv install 3.10
    pyenv global 3.10
    ```

    The code above will install latest Python 3.10, and activate this version globally for your machine.


## Install recommended tools to manage development environment

### Use a virtual environments to manage dependencies

Using virtual environments will allow you to isolate your Fed-BioMed development environment from your other projects. There, you have of course several options. You can use build-in python `venv` or `conda` or any other compatible Python virtual environment tools.


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
conda create --name fedbiomed-dev python=3.10
conda activate fedbiomed-dev
```


### Install Pdm

Once virtual environments are set `PDM` package manager can be installed via `PyPI`. It is also possible to install `PDM` globally for the Python version that is activated or base `conda` environment if `conda` is preferred as virtual environment manager.

```
pip install pdm
```

The command above will install PDM in the active `conda` environment or in active `pyenv` if it is used for managing Python versions.


### NodeJS and yarn

Fed-BioMed Node GUI uses the ReactJS library for web development. The ReactJS application for the Fed-BioMed Node GUI requires `Node.js` and `Yarn` to be installed, both for development and to enable automatic building during the creation of the Fed-BioMed package.

Therefore, it is essential to have Node.js and Yarn installed on your system. Please follow the installation instructions for your operating system to install [Node.js](https://nodejs.org/en/download) and [Yarn](https://classic.yarnpkg.com/en/).

## Using PDM

PDM is a tool to manage dependencies of a Python project, and build the python package:

- from `pyproject.toml` containing package source configuration, that is to say explicit instructions for build settings and dependencies
- from `pdm.lock` containing package locked cache version, that is to say a precise resolution of the configuration in `pyproject.toml`
Both are located in the root directory of the project.

Once PDM is installed, dependencies can be installed through the following command. It uses the existing `pdm.lock` to install dependencies for all Fed-BioMed components:

```
pdm install
```

This command will automatically detect the environment that used currently and install all dependencies in it. For example, if conda env is activated and there is no `.venv` folder in root directory of the project, all dependencies will  be installed within the `conda` environment. However, if another virtual environment is activated, or if `.venv` folder is existing PDM will install dependencies within that environment.

If there is no virtual environment activated pdm will create a default virtual environment using python's `virtualenv` in `.venv`

To install dependencies for only some Fed-BioMed component (from `researcher`, `gui`, `node`) use:

```
pdm install -G researcher
```


### Re-generate `pdm.lock`

For some reason you may want to remove and re-generate a new `pdm.lock`.

If there is no `pdm.lock` lock file when running `pdm install`, it first resolves all the dependencies defined in `pyproject.toml` and create a `pdm.lock` lock file.

If running `pdm lock` it only generates a new lock file, but does not perform the installation.

!!! warning "Important"
    By default, the optional packages are not included in generated the `pdm.lock` file. These optional packages contain dependencies specific to each Fed-BioMed component. Use `pdm install -G :all` or `pdm install -G researcher,node,gui` or `pdm lock -G :all` to generate lockfile with dependencies for all components.

There are also other installation options such selection of specific groups or extra dependencies. Please see `pdm install --help` and `pyproject.toml` file for more information.


### What about instant changes in source code?

PDM automatically adds package source to your environment that makes `fedbiomed` module accessible from Python interpreter. It also handles instant changes in the source code. If a new module is added or a module is edited PDM will automatically make changes available within the environment currently used. It means there is no need to install or rebuild `fedbiomed` package.

!!! warning "Important"
    You should never install `fedbiomed` from `pip` in your development environment. This may interrupt to access current source code changes in your development environment.


### Add new module

New packages can be added through `pyproject.toml`, or using the command `pdm add`. Please see `pdm add --help` for detailed usage instructions.

Once a new package added via `pyproject.toml` it may be required to run `pdm lock -G :all` to resolve the dependencies, then, execute `pdm install` to update packages and install missing ones.


### For more

Please visit `pdm` documentation for more information and usage details.


## Post actions

To verify your installation please run `pytest tests` and `tox -r` to make sure there is no missing module in your environment.


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
# use the python environment for [development](../docs/developer/development-environment.md)
cd ${FEDBIOMED_DIR}/gui/ui
yarn start
```

After that if you go ``localhost:3000`` you will see same user interface is up and running for development.  When you change the source codes
in ``${FEDBIOMED_DIR}/gui/ui/src`` it will get dynamically updated on ``localhost:3000``.

Since Flask is already started in debug mode, you can do your development/update/changes for server side (Flask) in `${FEDBIOMED_DIR}/gui/server`. React part (ui) on development mode will call API endpoint from `localhost:8484`, this is why first you should start Flask server first.

After development/debugging is done. To update changes in built GUI, you need rebuild the React app. Afterwards,
you will be able to see changes on the ``localhost:8484`` URL which serve built UI files.

```shell
yarn build
fedbiomed node gui start
```

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



