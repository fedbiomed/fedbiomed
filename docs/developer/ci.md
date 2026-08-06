# Continuous integration

Fed-BioMed uses [GitHub Actions](https://github.com/fedbiomed/fedbiomed/actions)
for pull-request checks, scheduled compatibility testing, package validation,
Docker testing, and releases.

The supported Python range is Python 3.11 through Python 3.14. The CI strategy
uses two complementary levels:

- Fast pull-request gates test the oldest and newest supported Python versions.
- Scheduled jobs test every supported Python version on the complete runner
  matrix.

This keeps pull-request feedback reasonably fast while still detecting
compatibility problems in intermediate Python versions.

## Runner labels

Workflow files use the following runner labels:

| Label | Ownership | Purpose |
| --- | --- | --- |
| `ubuntu-latest` | GitHub-hosted | Primary Linux pull-request and compatibility runner |
| `ubuntu-24-04` | Self-hosted | Project Ubuntu runner, including privileged Docker tests |
| `macos-latest` | GitHub-hosted | Primary macOS pull-request and compatibility runner |
| `macos-m1` | Self-hosted | Apple Silicon compatibility runner |

`ubuntu-latest` and `ubuntu-24-04` are not aliases for the same runner.
`ubuntu-latest` creates an ephemeral GitHub-hosted virtual machine, while
`ubuntu-24-04` is a project-defined label for a self-hosted runner.

## Workflow ownership

All workflow definitions are under `.github/workflows`.

| Workflow file | Responsibility | Triggers |
| --- | --- | --- |
| `build-test.yml` | Pull-request gate for unit tests, the MNIST test, and documentation | Non-draft pull requests targeting `develop` or `master` |
| `fbm-generic-test.yml` | Base implementation, used by the other workflows for documentation, unit, MNIST, and ordinary E2E jobs; also provides the configurable manual test UI | Called by other workflows or started manually |
| `python-compatibility.yml` | Complete test across all Python and runner (OS) versions on for unit and E2E tests (excet Endurance) | Sunday at 00:37 UTC |
| `end-to-end.yml` | Ordinary E2E testing after changes reach `master`, with optional manually supplied JSON matrices | Push to `master` or manual |
| `endurance-tests.yml` | Long-running endurance tests on the Python endpoints | Saturday at 01:00 UTC or manual |
| `package-compatibility.yml` | Builds one wheel and source distribution, then installs and checks the exact wheel across the supported matrix | Monday at 01:17 UTC, manual, or called by the release workflow |
| `test-docker.yml` | Tests if public docker images can be build for all python versions, and then checks VPN functional test by running the MNIST training across node and researcher images | Pull requests to `develop`, push to `master`, Monday at 01:17 UTC, or manual |
| `deploy.yml` | Validates the release package for Python wheel of Fedbiomed, publishes it to PyPI, and creates the GitHub release | Tag push |
| `docker-deploy.yml` | Builds public base, node, and researcher docker images and publishes those release images to Docker Hub | Version tag, or manual |
| `build-and-deploy-documentation.yml` | Builds versioned documentation and updates the public documentation repository | Tag push or manual |
| `codespell.yml` | Checks repository spelling and annotates errors | Pull requests targeting `develop` or `master` |
| `runner-maintenance.yml` | Bounded cleanup of the pip cache, Homebrew downloads, cached interpreters, and end-to-end datasets on every self-hosted runner | Sunday at 04:00 UTC or manual |

The workflow filename identifies the owner of a CI lane. The reusable
`fbm-generic-test.yml` file owns the test implementation, while small caller
workflows such as `build-test.yml` and `python-compatibility.yml` define when
and with which matrix it runs.

## Test strategy

### Pull-request gate

`build-test.yml` runs only for non-draft pull requests targeting `develop` or
`master`. It calls `fbm-generic-test.yml` with:

- Python 3.11 and Python 3.14
- `ubuntu-latest` and `macos-latest`
- the complete unit-test suite
- the MNIST E2E smoke test
- one documentation build on Python 3.11 and `ubuntu-latest`

The endpoint versions provide early warning for both the oldest and newest
supported interpreters. Intermediate versions are covered by the scheduled
matrix.

Superseded runs for the same pull request are cancelled. Pushing another commit
to the pull-request branch therefore replaces an obsolete run.

### Weekly Python compatibility

`python-compatibility.yml` runs once per week and calls
`fbm-generic-test.yml` with:

- Python 3.11, 3.12, 3.13, and 3.14
- `ubuntu-latest`, `ubuntu-24-04`, `macos-latest`, and `macos-m1`
- unit tests
- ordinary E2E tests

Ordinary E2E tests are sharded by test file. Each file receives its own job,
tox environment, timeout, logs, and dependency snapshot. A failure in one file
does not prevent unrelated shards from reporting their results.

Datasets are not sharded. `get_data_folder` caches them under
`~/.cache/fedbiomed/e2e-data`, so a self-hosted runner fetches MNIST, MedNIST,
and IXI once and later jobs reuse them; a hosted runner starts empty each time.
`FEDBIOMED_E2E_DATA_PATH` overrides that root and no workflow sets it.
`runner-maintenance.yml` prunes the cache.

The matrix uses `fail-fast: false`, so all Python and runner combinations can
report compatibility failures in a single workflow run.

### E2E testing on `master`

`end-to-end.yml` retains ordinary E2E coverage for pushes to `master`. Its
automatic default is Python 3.11 and 3.14 across the four compatibility
runners.

For a manual run, `python-version` and `os` accept JSON arrays. For example:

```text
python-version: ["3.12"]
os: ["ubuntu-latest"]
```

This workflow does not collect endurance files. Ordinary E2E environments
select `e2e_*.py` and exclude `endurance_*.py`.

Runs are superseded per ref. A push to `master` cancels an E2E run still in
flight for the same branch, so consecutive pushes do not stack multi-hour jobs
on the self-hosted queue. E2E therefore reports for the current tip of
`master`, not for every intermediate commit.

### Endurance testing

`endurance-tests.yml` is deliberately separate from ordinary E2E testing. It
runs once per week with:

- Python 3.11 and Python 3.14
- `ubuntu-latest` and `ubuntu-24-04`
- only `endurance_*.py`
- a six-hour job timeout
- explicit process-group cleanup

It has no pull-request or branch-push trigger because endurance tests are too
expensive for frequent development feedback.

## Reusable test workflow

`fbm-generic-test.yml` is the central implementation for the regular Python
test lanes. Callers provide:

- `python-versions`: JSON array of Python versions
- `os-list`: JSON array of runner labels
- `run-docs`: enable the documentation build
- `run-unit`: enable unit tests
- `run-mnist`: enable the MNIST smoke test
- `run-e2e`: enable the ordinary E2E shards

It creates exact tox environment names such as:

```text
py3.11-unit
py3.14-e2e-mnist
py3.12-e2e
```

Using an exact environment ensures that a matrix job cannot accidentally run
tests under a different interpreter.

The workflow uploads failure diagnostics for the enabled suite, including:

- JUnit XML reports
- tox logs
- E2E output
- test logs
- coverage XML where applicable
- the installed dependency snapshot

Unit-test coverage is uploaded to
[Codecov](https://app.codecov.io/gh/fedbiomed/fedbiomed/). A Codecov upload
failure does not fail the test job.

### Manual test selection

The Actions page exposes `fbm-generic-test.yml` as **Fed-BioMed Tests
(Reusable)**. Its manual form takes the same inputs as a workflow call:

- `python-versions`: JSON list, for example `["3.11","3.14"]`
- `os-list`: JSON list of runner labels, drawn from `ubuntu-latest`,
  `ubuntu-24-04`, `macos-latest`, and `macos-m1`
- `run-docs`, `run-unit`, `run-mnist`, `run-e2e`: one checkbox each

Both lists must be valid JSON arrays; the matrices consume them through
`fromJSON`, so a malformed value fails the run before any job starts. An empty
array produces no jobs for that matrix. Documentation ignores both lists
because it has a fixed Python and runner configuration.

## Python setup

`.github/actions/setup-fbm-env/action.yml` installs the interpreter requested by
the matrix and exports its resolved executable as `FEDBIOMED_PYTHON_BIN`.

The action handles:

- self-hosted macOS with Homebrew
- runners providing `dnf`, such as Fedora and its derivatives
- every other runner with `actions/setup-python`
- exact interpreter-version validation
- CPU-only PyTorch selection on Linux
- tox installation inside an isolated virtual environment

The isolated tox environment avoids modifying Homebrew-managed Python
installations and avoids the PEP 668 `externally-managed-environment` error.

On Linux the action exports
`PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`, which replaces about
3.8 GB of CUDA runtime wheels with a 190 MB CPU build. The tests never run on a
GPU: the node selects a CUDA device only when it is started with `--gpu` or
`--gpu-only`, and no test does that. Exporting `PIP_EXTRA_INDEX_URL` before the
action runs keeps the CUDA wheels. Other platforms are unaffected, because
PyTorch publishes no CUDA build for them.

## Package compatibility and releases

`package-compatibility.yml` separates building a package from testing its
installation:

1. Build one wheel and one source distribution on Python 3.11.
2. Validate their metadata with Twine.
3. Upload the build output as the `fedbiomed-package` artifact.
4. Download the same wheel into each Python and runner job.
5. Install it in a clean virtual environment.

The installation matrix covers Python 3.11 through Python 3.14 on all four
compatibility runners. It verifies:

- `pip check`
- `fedbiomed --help`
- the interpreter actually used by the environment
- `Requires-Python` and package extras
- the `fedbiomed` console-script entry point
- notebooks, tutorials, and common environment files under `SHARE_DIR`
- compiled React assets
- that imports come from the installed wheel rather than the checkout

`deploy.yml` calls this workflow for a tag. PyPI publication and GitHub release
creation depend on the tested package artifact, so the published files are the
same files that passed compatibility testing.

## Docker strategy

Docker testing is split into build smoke tests and a functional VPN test.

### Public-image build smoke

The `hosted-build-smoke` job in `test-docker.yml` runs on a fresh
`ubuntu-latest` runner for Python 3.11 through Python 3.14. It:

- builds `docker/base/Dockerfile`
- builds `docker/node/Dockerfile`
- builds `docker/researcher/Dockerfile`
- checks the selected Python inside each image
- checks the Fed-BioMed command entry points
- removes only the images created by that job

These images are test-only and are not pushed.

### VPN functional test

The `vpn-functional` job in `test-docker.yml` runs Python 3.11 and 3.14 on
GitHub-hosted and self-hosted Ubuntu. It builds the VPN server, researcher,
node, and GUI images and then:

- creates the WireGuard network
- connects a researcher and two nodes
- registers datasets on both nodes
- converts notebook 101 to a Python script
- runs a federated training experiment

The test uses run-specific image tags, Compose project names, container names,
and network names. The matrix is currently serial because the VPN environment
uses fixed host ports.

Compatibility CI explicitly selects the non-GPU node base and CPU-only PyTorch
to fit within CI disk limits. This is an opt-in test configuration:

- `FBM_VPN_NODE_BASE_SERVICE=basenode-no-gpu`
- `FBM_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`

Normal VPN deployment does not set these values. It keeps the GPU-capable node
base and standard package-index resolution.

The CPU PyTorch installation is created in the VPN base and reused by the
researcher, non-GPU node base, and GUI images. The researcher and node package
builds skip the React build because neither image serves the node GUI. Node.js,
Yarn, and the React compilation remain in the dedicated GUI image.

On the self-hosted runner, useful Docker layers persist between jobs. A bounded
prune before and after the VPN test targets 8 GB of retained build cache
without deleting Docker images, containers, volumes, or networks. Hosted
runners are ephemeral and do not need persistent cache management.

The VPN build wrapper propagates the first failed Docker build instead of
continuing with later images. Cleanup removes resources created by the current
run and does not execute a broad `docker system prune`.

### Docker publication

`docker-deploy.yml` owns publication of the public base, node, and researcher
images. Published images use one Python runtime, currently Python 3.14. It is
the default `PYTHON_VERSION` in `docker/base/Dockerfile`; `docker-deploy.yml`
passes no build argument, so editing that default changes what is published.

A `v*.*.*` tag publishes the generated tags to Docker Hub. Manual runs build
the same images without publishing them. Image builds are otherwise covered by
`test-docker.yml`, which builds the three Dockerfiles on every pull request to
`develop`.

## Schedule

GitHub cron expressions use UTC and have five fields:

```text
┌──────── minute
│ ┌────── hour
│ │ ┌──── day of month
│ │ │ ┌── month
│ │ │ │ ┌ day of week, where Sunday is 0
│ │ │ │ │
37 0 * * 0
```

| Cron | Workflow | Meaning |
| --- | --- | --- |
| `0 1 * * 6` | `endurance-tests.yml` | Saturday at 01:00 UTC |
| `37 0 * * 0` | `python-compatibility.yml` | Sunday at 00:37 UTC |
| `17 1 * * 1` | `package-compatibility.yml` | Monday at 01:17 UTC |
| `17 1 * * 1` | `test-docker.yml` | Monday at 01:17 UTC |

Scheduled workflows always execute from the repository's default branch.
Changing a cron entry on a feature branch does not make that schedule active
until the change reaches the default branch.

## Inspecting and rerunning CI

To inspect a failure:

1. Open the pull request or the repository **Actions** page.
2. Select the workflow run.
3. Select the failing matrix job.
4. Inspect the first failing step rather than only the final cleanup error.
5. Download the diagnostic artifact when one is available.

Pushing a new commit reruns pull-request workflows. A completed job can also be
rerun from its workflow page.

You can use the `fbm-generic-test.yml` workflow to run a manual job for a targeted Python/test/runner
combination instead of rerunning the complete scheduled matrix.

## Self-hosted runner requirements

Self-hosted runners must provide the tools required by their assigned jobs:

- a supported shell and Git
- passwordless access for the package-manager commands used by the setup action
- enough disk space for tox environments and test artifacts
- Docker and Docker Compose for Docker-assigned Ubuntu runners
- `/dev/net/tun` and the capabilities required by the VPN functional test
- Homebrew on the self-hosted Apple Silicon runner

Runner labels are part of the workflow interface. If a runner is renamed or
relabelled, update every matrix that refers to its old label.

GitHub cannot address every self-hosted runner in one job, so
`runner-maintenance.yml` lists them individually. A new runner is cleaned only
once its label is added to that matrix. A runner that is offline when the
workflow starts keeps its job queued until GitHub cancels it, so a failing leg
means the machine needs attention.
