
## Self-Hosted Runner conf

### Pyenv and python version installation

Workflows use `setup-pyenv/action.yml` to install `pyenv`. `pyenv` allows to install requires python versions.

This action installs `pyenv` in Linux and MacOs systems. However, there may be some missing packages in the ci slaves. Therefore, please carefully check `pyenv` installation and python version installation steps to make sure all required packages are installed in the system.

The error while installation specific python version can be difficult to debug on action output. Please connect self-hosted ci and execute installation manually (e.g pyenv install 3.10)

One of the known error on Fedora is a missing C++ compiler tool: gcc, solution is it install `gcc` `yum install gcc`

### Permissions

The runner user should have sudo access and be able to execute installation commands without needing to enter a password. Please modify the `sudoers` file by running sudo visudo and add `ci ALL=(ALL:ALL) NOPASSWD: /usr/bin/apt-get` or `ci ALL=(ALL:ALL) NOPASSWD: /usr/bin/dnf` at the end of the file. This will allow the runner user to install necessary packages during execution.


### Setup Python: action/setup-python self-hosted runner configuration

Default github action for installing python version has some problems on self-hosted runners. Please visit https://github.com/actions/setup-python/blob/2f078955e4d0f34cc7a8b0108b2eb7bbe154438e/docs/advanced-usage.md#using-setup-python-with-a-self-hosted-runner to find our more about how to conifgure self hosted runner for setup python.

Important information: Env variables created in this document has to be add in `actions-runner/.env` file.




