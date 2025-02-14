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
