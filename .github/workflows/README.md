
## Self-Hosted Runner conf

### Python version installation

Test workflows install their interpreter through `.github/actions/setup-fbm-env`.
The action picks a method from what the runner provides:

- Homebrew, on the self-hosted Apple Silicon runner
- `dnf`, on Fedora and derivatives
- `actions/setup-python`, everywhere else

It then resolves the real interpreter path, validates that the installed version
is exactly the one requested, exports it as `FEDBIOMED_PYTHON_BIN`, selects
CPU-only PyTorch on Linux, and installs tox into an isolated virtual
environment. The isolated tox environment avoids modifying a Homebrew-managed
Python and avoids the PEP 668 `externally-managed-environment` error.

Interpreter installation failures are hard to read in the action output. Connect
to the self-hosted runner and install the version manually to see the real
error, for example `brew install python@3.14` or `sudo dnf install python3.14`.

A known Fedora failure is a missing C++ compiler; install it with
`sudo dnf install gcc`.

### Permissions

The runner user should have sudo access and be able to execute installation commands without needing to enter a password. Please modify the `sudoers` file by running sudo visudo and add `ci ALL=(ALL:ALL) NOPASSWD: /usr/bin/apt-get` or `ci ALL=(ALL:ALL) NOPASSWD: /usr/bin/dnf` at the end of the file. This will allow the runner user to install necessary packages during execution.

### Install Necessary Packages 

#### Ubuntu
Please install floowing packages into self hosted runner.
```
sudo apt update
sudo apt install -y python3-tk \
    gcc libgmp3-dev libmpfr-dev libmpc-dev \
    zlib1g libjpeg-dev libpng-dev libssl-dev vim nodejs
```

Install yarn 

```
npm install -g yarn
```
##### Install Docker

Please also fllow docker installation instructions https://docs.docker.com/engine/install/ubuntu/

### Setup Python: action/setup-python self-hosted runner configuration

Default github action for installing python version has some problems on self-hosted runners. Please visit https://github.com/actions/setup-python/blob/2f078955e4d0f34cc7a8b0108b2eb7bbe154438e/docs/advanced-usage.md#using-setup-python-with-a-self-hosted-runner to find our more about how to conifgure self hosted runner for setup python.

Important information: Env variables created in this document has to be add in `actions-runner/.env` file.