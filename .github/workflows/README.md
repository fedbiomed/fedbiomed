
## Self-Hosted Runner conf

The Apple Silicon machine labelled `macos-m1` is the only self-hosted runner in
the matrices. Every other job runs on a GitHub-hosted runner.

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
error, for example `brew install python@3.14`.

### Node.js and Yarn

The jobs that build the node GUI install their own toolchain through
`.github/actions/setup-node-env`, which provisions Node with nvm and Yarn with
npm on a self-hosted runner. Nothing has to be installed by hand, but the nvm
initialisation it appends to `~/.bash_profile` must survive between jobs.

### Setup Python: action/setup-python self-hosted runner configuration

Default github action for installing python version has some problems on self-hosted runners. Please visit https://github.com/actions/setup-python/blob/2f078955e4d0f34cc7a8b0108b2eb7bbe154438e/docs/advanced-usage.md#using-setup-python-with-a-self-hosted-runner to find our more about how to conifgure self hosted runner for setup python.

Important information: Env variables created in this document has to be add in `actions-runner/.env` file.