# Configuring Development Environment

This article is written to guide developers to create development environment. This guide can also be used to build Fed-BioMed software from its source.

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

Before you start using Fed-BioMed or developing on Fed-BioMed please make sure that required Python version 
is installed on your machine. 
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
conda create --name fedbiomed-dev python=3.11
conda activate fedbiomed-dev
```


### Install Pdm

Once virtual environments are set `PDM` package manager can be installed via `PyPI`. It is also possible to install `PDM` globally for the Python version that is activated or base `conda` environment if `conda` is preferred as virtual environment manager.

```
pip install pdm
```

The command above will install PDM in the active `conda` environment or in active `pyenv` if it is used for managing Python versions.


### NodeJS and Yarn

Node.js and Yarn are required for frontend development and for building the
GUI distribution from source. The current GUI build hook runs `yarn install` and `yarn build`, including during editable installation unless explicitly skipped using `FBM_SKIP_FRONTEND_BUILD=1`.
Core and API builds do not require frontend tools. Installing a released GUI wheel uses its bundled frontend assets and does not require Yarn, unless the user explicitly commands to rebuild the GUI.

See [Node.js](https://nodejs.org/en/download) and
[Yarn Classic](https://classic.yarnpkg.com/en/) for installation instructions.

## Developer Installation Using PDM

### Projects, workspace, and dependency groups

Run installation and locking commands from the repository root. There are three
package definitions: root `pyproject.toml` for `fedbiomed`,
`fedbiomed_node_api/pyproject.toml` for the API, and
`fedbiomed_gui/pyproject.toml` for the GUI. Development uses the root `pdm.lock`
and one selected environment. The package dependency chain is
`fedbiomed-gui` → `fedbiomed-node-api` → `fedbiomed`.

For developers, if you want to use pdm to manage dependencies, and use the new experimental feature for the workspace configuration across multiple packages while keeping a single lock file, you should ensure that you have the PDM version 2.28 or above. You can check it with
`pdm --version`; this repository does not currently pin a tested PDM version. If you do not wish to use this experimental feature, with your preferred environment manager, you can just use the 3 separate `pyproject.toml` files to install the dependencies.  

The workspace tells PDM where to find the API and GUI packages in this repository.
It is defined in the root `pyproject.toml`, with member paths relative to the
repository root:

```toml
[tool.pdm.workspace]
members = ["fedbiomed_node_api", "fedbiomed_gui"]
```

The `local` group installs both packages in editable mode, so Python source
changes are available without reinstalling. For example, `pdm sync -G local`
installs the GUI even without `-G gui`. Check the groups in `pdm.lock` to see
which packages each group selects, and use `pdm list` to see what is installed.
Selecting fewer groups does not automatically remove packages installed earlier.
See the [PDM workspace documentation](https://pdm-project.org/en/latest/usage/workspace/).

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
`local` uses `[tool.pdm.dev-dependencies]`; the other development groups use
`[dependency-groups]`.

### (Optional) Using PDM to manage your virtual Python environment

PDM records its selected interpreter in `.pdm-python`. Activating another shell
environment does not necessarily replace that selection. Check and switch it
explicitly:

```sh
pdm venv list
pdm info --python
pdm venv create --name myenv 3.11
pdm use --venv myenv
pdm info --python
```

The named environment leaves an existing `.venv` intact. To return to that
project environment, run `pdm use .venv/bin/python`. Without a saved interpreter,
PDM can discover an existing environment or create `.venv`; do not rely on
shell activation alone to identify the environment used by PDM.
See [PDM environment selection](https://pdm-project.org/en/latest/usage/venv/).

### Install and select groups

With an existing lockfile, `pdm install` without group-selection options installs
all groups recorded in that lockfile. This means all components only when their
groups are locked; it is not an unconditional selection of every optional extra.
`pdm install` can also create or update the lockfile when needed.

`pdm install -G researcher` selects core and researcher dependencies along with
the default development-group selection; it does not mean researcher alone.
`--prod` excludes development groups. Explicitly naming a development group
selects that group instead of all development groups. Refer to
[PDM group selection](https://pdm-project.org/en/latest/usage/dependency/#select-a-subset-of-dependency-groups-to-install).

For development of all three packages and their tests, use the checked-in lockfile:

```sh
pdm lock --check
pdm sync -G local -G gui -G researcher -G test -G dev
```

`local` includes both siblings, even when only one is needed for a particular
change. Published package metadata retains regular versioned requirements;
local paths are development configuration only. The GUI installation runs its
frontend build hook. If built assets already exist, prefix `pdm sync` with
`FBM_SKIP_FRONTEND_BUILD=1`. Python-only tests that supply their own assets may
also skip that build, but this does not produce a working frontend bundle.

For API-only installation from a lockfile containing the local API package in
`node-api` and the GUI only in `gui`/`local`, use `pdm sync --prod -G node-api`.
Check this in a fresh environment: selecting fewer groups does not remove
packages installed earlier. Workspace changes or regenerating the lockfile can
change group membership, so inspect the lockfile and installation when validating
package separation.

### Install, synchronize, and lock

| Command | Effect |
| --- | --- |
| `pdm install` | Checks the lockfile, resolves if needed, and installs selected dependencies. |
| `pdm sync` | Installs selected dependencies from the existing lockfile without regenerating it. |
| `pdm lock` | Resolves dependencies and updates the lockfile without installing them. |

Do not regenerate the lockfile for every fresh environment. After changing
package dependencies or versions, run:

```sh
pdm lock -G :all --update-reuse
pdm lock --check
pdm sync -G local -G gui -G researcher -G test -G dev
```

Review and commit `pdm.lock` together with the changed package definitions.
`--update-reuse` attempts to retain existing locked versions. If a selected group
is missing from lockfile metadata, regenerate the lockfile rather than editing
its group labels or content hash by hand.

### Verify a fresh installation

After creating and selecting a fresh environment and syncing the groups
above, verify imports and installed distribution versions:

```sh
pdm run python -c 'import sys; print(sys.executable)'
pdm run python -c 'from importlib.metadata import version; print({n: version(n) for n in ("fedbiomed", "fedbiomed-node-api", "fedbiomed-gui")})'
```

The python path should point to your newly created environment, and all three versions of the imported packages must match.

### What about instant changes in source code?

PDM automatically adds package source to your environment that makes `fedbiomed` module accessible from Python interpreter. It also handles instant changes in the source code. If a new module is added or a module is edited PDM will automatically make changes available within the environment currently used. It means there is no need to install or rebuild `fedbiomed` package.

!!! warning "Important"
    You should never install `fedbiomed` from `pip` in your development environment. This may interrupt to access current source code changes in your development environment.


### Add new module

New packages can be added through `pyproject.toml`, or using the command `pdm add`. Please see `pdm add --help` for detailed usage instructions.

Once a new package added via `pyproject.toml` it may be required to run `pdm lock -G :all` to resolve the dependencies, then, execute `pdm install` to update packages and install missing ones.


### For more

Please visit `pdm` documentation for more information and usage details.


### Post actions

Use the focused installation checks in [Verify a fresh installation](#verify-a-fresh-installation). Run `pdm run python -m pytest tests` for the wider suite; tox validates separate test environments.


## Development/Debugging for GUI

Start the GUI and API together using either command:

```shell
fedbiomed-gui --path /path/to/my-node
# Equivalent core command:
fedbiomed node --path /path/to/my-node gui start
```

Both serve the built frontend at `http://localhost:8484`. The data folder defaults
to the `data` directory in the node's root; use `--data-folder /alternate/data-path`
to select another directory.

If frontend assets are missing or you have changed the frontend sources, add
`--recreate` to either command:

```shell
fedbiomed-gui --path /path/to/my-node --recreate
fedbiomed node --path /path/to/my-node gui start --recreate
```

This runs `yarn install` and `yarn build` once before serving the generated files.
It requires frontend sources and Node.js/Yarn. Later source edits require another
rebuild.

For automatic frontend updates while editing, start the API separately using
either syntax:

```shell
fedbiomed-node-api --path /path/to/my-node
# Equivalent core command:
fedbiomed node --path /path/to/my-node api start

# Optional alternate data directory:
fedbiomed-node-api --path /path/to/my-node --data-folder /alternate/data-path
```

Neither `--development` nor `--debug` is required for this workflow.
`--development` selects Flask's development server instead of Gunicorn.
`--debug` enables Flask debug mode and debug-level application logging; combined
with `--development`, it also enables the debugger and backend code reloader.
For backend development, for example:

```shell
fedbiomed-node-api --path /path/to/my-node --development --debug
```

In another terminal, start React's development server:

```shell
cd ${FEDBIOMED_DIR}/fedbiomed_gui/ui
yarn start
```

After that if you go ``localhost:3000`` you will see same user interface is up and running for development.  When you change the source codes
in ``${FEDBIOMED_DIR}/fedbiomed_gui/ui/src`` it will get dynamically updated on ``localhost:3000``.

Since Flask is already started in debug mode, you can do your development/update/changes for server side (Flask) in `${FEDBIOMED_DIR}/fedbiomed_gui/server`. React part (ui) on development mode will call API endpoint from `localhost:8484`, this is why first you should start Flask server first.

After development/debugging is done, rebuild the React app to see the changes on the ``localhost:8484`` URL, which serves
the built UI files. `--recreate` rebuilds it before starting the server:

```shell
fedbiomed-gui --path /path/to/my-node --recreate
```
(or in the previous versions)
```shell
fedbiomed node gui start --recreate
```

## Building the distributions

Core, Node API, and GUI have separate build configurations. From the repository
root, build the core and API wheels and source archives with:

```sh
pdm build
pdm build -p fedbiomed_node_api
```

Core includes `fedbiomed` and its existing shared resources; the API includes
`fedbiomed_node_api` and its configuration template; the GUI includes
`fedbiomed_gui` and compiled frontend assets. Before packaging the GUI, explicitly
build its frontend:

```sh
cd fedbiomed_gui/ui
yarn install --frozen-lockfile
yarn build
cd ../..
pdm build -p fedbiomed_gui
```

Packaging never invokes Yarn. Both GUI artifacts contain the existing bundle;
the source archive also contains frontend sources. Rebuilding a released source
archive requires no Node.js/Yarn. Missing `index.html` or JavaScript files cause
packaging to fail, even if `FBM_SKIP_FRONTEND_BUILD=1` is set. Rebuild frontend
assets explicitly after editing their sources.

The dependency chain is `fedbiomed-gui` → `fedbiomed-node-api` → `fedbiomed`.
Core's `node-api` and `gui` extras select the corresponding distributions;
the web dependencies belong to the API. All three versions must match,
with exact dependency pins. When updating the version, update core's
`fedbiomed/__init__.py`, both sibling `pyproject.toml` versions, and the pins
in all three project files. Packaging tests check their consistency.

For source development, PDM resolves the siblings from the
development-only `local` group as editable packages. From the repository root:

```sh
pdm sync -G gui -G local -G test
```

Use the checked-in lockfile for installation. After changing package versions
or dependencies, regenerate it with `pdm lock -G :all --update-reuse`, then
sync again. Keep the editable sibling packages selected during API/GUI
development so imports use this checkout.

Editable installation does not require frontend assets or invoke Yarn.
Use `fedbiomed-gui --recreate` to build the frontend before serving it.
Editable Python changes are available immediately; frontend changes still require rebuilding.
Local paths are confined to development configuration and the lockfile;
wheel dependency metadata uses regular package names and version constraints.

To check package contents and rebuild wheels from source archives, use an
environment with `pytest` and `hatchling` installed:

```sh
python -m pytest --noconftest -c /dev/null tests/test_package_builds.py -q
```

These packaging checks run without application dependencies or frontend tools.

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
### Validation for manually installing from wheel

If you are manually installing from python wheels instead of using the latest versions from pip, we have a script to check if the wheels you are trying to install are compatible with each other. See, the [package compatibility workflow](ci.md#package-compatibility-and-releases) for details on using this validation script. It checks all three wheels and source archives, then tests isolated installations of core, API, GUI, and researcher profiles. 

### Building Fed-BioMed Takes too Long

Some static files located in the root Fed-BioMed source directory (e.g., notebooks, tests, etc.) are also included in the final distribution. Therefore, having large data files or artifacts left from operations for testing and development purposes can increase the loading time. Please ensure that such data files are cleared before building the Fed-BioMed package to reduce build time.
