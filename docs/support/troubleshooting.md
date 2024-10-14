# Troubleshooting and FAQ

This page contains some frequently encountered problems, and their solutions.

---

## Tkinter Error on macOS or Missing `python-itk` Package

Fed-BioMed uses the `itk` module in its CLI to launch a file explorer for selecting files and folders. However, some systems or Python environment managers may not include `python-itk`, which can cause failures—for example, when adding datasets into nodes via the Fed-BioMed CLI. If you're using tools like `pyenv` to install different Python versions without `python-itk` being present on your system, this will result in a Python installation without the correct `itk` module.

You can verify if `itk` or `python-itk` is correctly installed by running the following test code:

```python
import tkinter
tkinter._test()
```

If the necessary modules are not installed, you may encounter an exit error on macOS similar to this:

```
macOS 14 (1407) or later required, have instead 14 (1406)!
```

If this occurs, you can install `python-itk` by running `brew install python-itk`

After installing, reinstall the Python version using `pyenv` to ensure the correct setup.

This issue may also occur if you're using a `conda` virtual environment. To ensure `itk` is correctly installed in `conda`, run:

```bash
conda install -c conda-forge itk
```
---

## Tkinter error on Linux

Similar to macOS, on Linux you may encounter errors due to the lack of the tkinter module installed on your local machine. On Linux systems, please run sudo apt-get install python3-tk before creating a pyenv for a specific Python version, or before using any other virtual environment tool.

On Fedora, please use sudo dnf install tk-devel.

## Missing `gmp.h`, `mpfr.h` and `mpc.h`

Some Fed-BioMed secure aggregation modules uses `gmpy2` for big integer operation. This module requires to have `libgmp3-dev` and `libmpfr-dev` and `libmpc-dev` installed on Linux debian distributions (and equivalents on different Linux distributions). In case of missing `gmp.h`,  `mpfr.h`, or `mpc.h` module errors please install `apt-get install libgmp3-dev libmpfr-dev libmpc-dev`.

On maxOs please install: `brew link gmp mpfr mpc`.
