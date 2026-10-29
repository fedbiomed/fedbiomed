# Troubleshooting and FAQ

This page contains some frequently encountered problems, and their solutions.

---

## Missing Python interpreter in Jupyter notebook

If Fed-BioMed is installed in a virtual environment such as `virtualenv` or `conda`, executing `fedbiomed researcher -p <component-dir>` may not use the correct Python interpreter. This issue can be resolved by installing the necessary packages for the virtual environment tool, either automatically or by manually registering a Python kernel.

For Conda users, installing `nb_conda_kernels` will automatically detect the Python interpreters from created virtual environments and allow you to select them.

For other tools, you may need to install `ipykernel` and register the Python interpreter manually. After activating your virtual environment, execute the following command:
```bash
python -m ipykernel install --user --name=<project-kernel-name>
```
This will make the Python environment used by the virtual environment selectable in Jupyter Notebook.

Another option is to add the Python path to the PATH variable:
```bash
export PATH=<path/to/python3>:$PATH
```

## Tkinter Error on macOS or Missing `python-tk` Package

Fed-BioMed uses the `itk` module in its CLI to launch a file explorer for selecting files and folders. However, some systems or Python environment managers may not include `python-tk`, which can cause failures—for example, when adding datasets into nodes via the Fed-BioMed CLI. If you're using tools like `pyenv` to install different Python versions without `python-tk` being present on your system, this will result in a Python installation without the correct `tk` module.

You can verify if `itk` or `python-tk` is correctly installed by running the following test code:

```python
import tkinter
tkinter._test()
```

If the necessary modules are not installed, you may encounter an exit error on macOS similar to this:

```
macOS 14 (1407) or later required, have instead 14 (1406)!
```

If this occurs, you can install `python-tk` by running `brew install python-tk`

After installing, reinstall the Python version using `pyenv` to ensure the correct setup.

This issue may also occur if you're using a `conda` virtual environment. To ensure `tk` is correctly installed in `conda`, run:

```bash
conda install -c conda-forge tk
```
---

## Tkinter error on Linux

Similar to macOS, on Linux you may encounter errors due to the lack of the tkinter module installed on your local machine. On Linux systems, please run sudo apt-get install python3-tk before creating a pyenv for a specific Python version, or before using any other virtual environment tool.

On Fedora, please use sudo dnf install tk-devel.

## Missing `gmp.h`, `mpfr.h` and `mpc.h`

Some Fed-BioMed secure aggregation modules uses `gmpy2` for big integer operation. This module requires to have `libgmp3-dev` and `libmpfr-dev` and `libmpc-dev` installed on Linux debian distributions (and equivalents on different Linux distributions). In case of missing `gmp.h`,  `mpfr.h`, or `mpc.h` module errors please install `apt-get install libgmp3-dev libmpfr-dev libmpc-dev`.

On macOS please install: `brew link gmp mpfr mpc`.

## Node Does Not Connect to the Researcher

To check whether a node is connected to the researcher, and why not, ask the node for the last connection state it recorded:

```bash
$ fedbiomed node status --history
```

For a node running in a Docker container, run the same command in the container:

```bash
$ docker exec -u fedbiomed <my-node-name> fedbiomed node -p /fbm-node status --history
```

Then read the `State` and `Reason` fields of the `Connection to researcher` table:

- `connected`: the node is connected; if experiments still do not reach it, check that the node shares a dataset with the tags the experiment requests.
- `disconnected` with *Researcher server at ... is not available; retrying.*: the node cannot reach the researcher. Check that the researcher is started, and that `Host` and `Port` match its address (`FBM_RESEARCHER_IP`, `FBM_RESEARCHER_PORT`, or the `[researcher]` section of `config.ini`) and are reachable from the node (firewall, Docker network).
- `failed`: the researcher refused the connection or the node refused the researcher, typically for a certificate or mutual authentication problem. Match the `Reason` with the symptoms listed in [mutual authentication](../user-guide/deployment/mutual-tls.md#verifying-and-troubleshooting).
- `Connection to researcher: none recorded yet`: the node has never been started.

If the node process is not running, the table shows the state the node was in when it stopped. See [Checking the connection to the researcher](../user-guide/nodes/configuring-nodes.md#checking-the-connection-to-the-researcher) for the description of every field.

## Debug Mode

Fedbiomed includes a debug mode, which when enabled prints more verbose outputs, including the caller function and line number in the output. The debug mode can be enabled separately on Node or Researcher, for component specific outputs, or can be enabled for both.

To enable per component, simply add `--debug` while starting the component, such as:

```bash
$ fedbiomed node start --debug
```

or

```bash
$ fedbiomed researcher start --debug
```

To enable for both components, set the environment variable `'FBM_DEBUG'` to `1, 'yes' or True`, and then start the component as usual.

```bash
$ export FBM_DEBUG=1
$ fedbiomed node start
```

The debug mode includes more detailed information in component initializations, training round starts and finishes, database operations (such as for datasets, training plans...) and the connection information between nodes-researcher and node-to-node.

## gRPC Logging

gRPC logs on its own, next to the Fed-BioMed logger, and the debug mode above does not raise it: `--debug` and `FBM_DEBUG` reach the Fed-BioMed logger only. Fed-BioMed sets `GRPC_VERBOSITY` to `ERROR` on the Node and on the Researcher alike, so gRPC prints errors only. It reports a failed TLS handshake at INFO, which a node retrying its connection repeats every couple of seconds on both ends.

A value set in the environment takes precedence. Raise it to read what gRPC itself reports, such as the OpenSSL reason a connection is refused:

```bash
$ GRPC_VERBOSITY=INFO fedbiomed node start
```

```bash
$ GRPC_VERBOSITY=INFO fedbiomed researcher start
```

gRPC reads this variable once, when it is imported, so it has to be set before the component starts. Diagnosing a connection that never establishes is covered in [mutual authentication](../user-guide/deployment/mutual-tls.md#verifying-and-troubleshooting).
