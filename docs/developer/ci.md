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
| `ubuntu-latest` | GitHub-hosted | Primary Linux pull-request runner |
| `ubuntu-24.04` | GitHub-hosted | Oldest supported Ubuntu, pinned for compatibility testing |
| `ubuntu-26.04` | GitHub-hosted | Newest supported Ubuntu, a preview image |
| `macos-latest` | GitHub-hosted | Primary macOS pull-request and compatibility runner |
| `macos-m1` | Self-hosted | Apple Silicon compatibility runner |

`ubuntu-latest` currently resolves to the same image as `ubuntu-24.04`. The
compatibility matrices name the version explicitly so a future move of the
`-latest` alias cannot change what they test.

GitHub publishes `ubuntu-26.04` as a preview image: it carries no availability
guarantee and its contents can change without the notice a stable image gets. A
job that stalls in the queue or fails only on that label should be checked
against [actions/runner-images](https://github.com/actions/runner-images) before
anything in this repository is suspected.

## Workflow ownership

All workflow definitions are under `.github/workflows`.

| Workflow file | Responsibility | Triggers |
| --- | --- | --- |
| `build-test.yml` | Unit tests, the MNIST test, and documentation: the endpoint interpreters on hosted runners for a pull request, every interpreter on every runner on a schedule | Non-draft pull requests targeting `develop` or `master`, and Monday to Friday at 18:00 UTC |
| `fbm-generic-test.yml` | Base implementation, used by the other workflows for documentation, unit, MNIST, and ordinary E2E jobs; also provides the configurable manual test UI | Called by other workflows or started manually |
| `end-to-end.yml` | Ordinary E2E testing, with optional manually supplied JSON matrices | Monday to Friday at 23:00 UTC, push to `master`, or manual |
| `endurance-tests.yml` | Long-running endurance tests on the Python endpoints | Saturday at 09:00 UTC or manual |
| `package-compatibility.yml` | Builds one wheel and source distribution, then installs and checks the exact wheel across the supported matrix | Monday at 03:00 UTC, manual, or called by the release workflow |
| `test-docker.yml` | Tests if public docker images can be build for all python versions, and then checks VPN functional test by running the MNIST training across node and researcher images | Monday to Friday at 20:00 UTC or manual |
| `deploy.yml` | Validates the release package for Python wheel of Fedbiomed, publishes it to PyPI, and creates the GitHub release | Tag push |
| `docker-deploy.yml` | Builds public base, node, and researcher docker images, and publishes them to Docker Hub when a version tag triggered the run | Version tag or manual |
| `build-and-deploy-documentation.yml` | Builds versioned documentation and updates the public documentation repository | Tag push or manual |
| `codespell.yml` | Checks repository spelling and annotates errors | Pull requests targeting `develop` or `master` |
| `runner-maintenance.yml` | Bounded cleanup of the pip cache, Homebrew downloads, cached interpreters, and end-to-end datasets on the self-hosted runner | Monday at 01:00 UTC or manual |

The workflow filename identifies the owner of a CI lane. The reusable
`fbm-generic-test.yml` file owns the test implementation, while small caller
workflows such as `build-test.yml` and `end-to-end.yml` define when
and with which matrix it runs.

## Test strategy

### Build test

`build-test.yml` owns the same three suites for both of its triggers, and only
the matrix changes between them. It calls `fbm-generic-test.yml` with the
complete unit-test suite, the MNIST E2E smoke test, and one documentation build
on Python 3.11 and `ubuntu-latest`.

For a non-draft pull request targeting `develop` or `master`:

- Python 3.11 and Python 3.14
- `ubuntu-latest` and `macos-latest`

Monday to Friday at 18:00 UTC:

- Python 3.11, 3.12, 3.13, and 3.14
- `ubuntu-24.04`, `ubuntu-26.04`, `macos-latest`, and `macos-m1`

The endpoint versions provide early warning for both the oldest and newest
supported interpreters on a pull request. Intermediate versions, the second
Ubuntu, and the self-hosted runner are covered by the scheduled run, within a
day.

Superseded runs for the same pull request are cancelled. Pushing another commit
to the pull-request branch therefore replaces an obsolete run. Scheduled runs
share a group keyed on the ref, so a run still in flight is replaced by the next
evening's.

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

### E2E testing

`end-to-end.yml` owns ordinary E2E coverage. It runs Monday to Friday at
23:00 UTC and on a push to `master`. Its automatic default is Python
3.11 and 3.14 across the four compatibility runners.

For a manual run, `python-version` and `os` accept JSON arrays. For example:

```text
python-version: ["3.12"]
os: ["ubuntu-24.04"]
```

This workflow does not collect endurance files. Ordinary E2E environments
select `e2e_*.py` and exclude `endurance_*.py`.

Runs are superseded per ref, so a scheduled run replaces one still in flight on
the same branch and consecutive pushes to `master` do not stack multi-hour jobs
on the self-hosted queue. E2E therefore reports for the current tip of a branch,
not for every intermediate commit.

### Endurance testing

`endurance-tests.yml` is deliberately separate from ordinary E2E testing. It
runs once per week, on Saturday morning, with:

- Python 3.11 and Python 3.14
- `ubuntu-24.04` and `ubuntu-26.04`
- only `endurance_*.py`
- a six-hour job timeout
- explicit process-group cleanup

It has no pull-request or branch-push trigger because endurance tests are too
expensive for frequent development feedback.

## Reusable test workflow

`fbm-generic-test.yml` is the central implementation for the regular Python
test lanes. Callers provide:

- `python-versions`: JSON array of Python versions, `["3.11","3.14"]` by default
- `os-list`: JSON array of runner labels, `["ubuntu-latest","macos-latest"]` by
  default
- `run-docs`: enable the documentation build
- `run-unit`: enable unit tests
- `run-mnist`: enable the MNIST smoke test
- `run-e2e`: enable the ordinary E2E shards

Every `run-` switch is off by default, so a caller lists only what it enables.

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
(Reusable)**. Its manual form takes the same inputs as a workflow call, but
defaults `os-list` to all four compatibility runners rather than the two
pull-request ones:

- `python-versions`: JSON list, for example `["3.11","3.14"]`
- `os-list`: JSON list of runner labels, drawn from `ubuntu-24.04`,
  `ubuntu-26.04`, `macos-latest`, and `macos-m1`
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
`ubuntu-24.04` and `ubuntu-26.04`. It builds the VPN server, researcher,
node, and GUI images and then:

- creates the WireGuard network
- connects a researcher and two nodes
- registers datasets on both nodes
- converts notebook 101 to a Python script
- runs a federated training experiment

The test uses run-specific image tags, Compose project names, container names,
and network names. Host ports are fixed, so two legs must never share a
machine. The runner topology provides that: every leg gets its own ephemeral
GitHub-hosted virtual machine. Moving a leg onto a self-hosted runner that
accepts more than one job at a time would break the assumption and cause port
collisions.

The same constraint applies to a developer machine. Only one VPN stack can run
at a time on a given host — `FBM_CONTAINER_INSTANCE_ID` distinguishes container
and network names, but not host ports.

Runs are superseded per ref, so a manual run replaces an in-flight scheduled one
instead of queueing behind it.

Compatibility CI explicitly selects the non-GPU node base and CPU-only PyTorch
to fit within CI disk limits. This is an opt-in test configuration:

- `FBM_VPN_NODE_BASE_SERVICE=basenode-no-gpu`
- `FBM_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`

Normal VPN deployment does not set these values. It keeps the GPU-capable node
base and standard package-index resolution.

`FBM_PYTORCH_INDEX_URL` reaches the researcher and node builds as a Docker
build argument and becomes an extra package index for their `pip install`, so
those images resolve the CPU-only PyTorch wheels. Left empty, the builds use the
default index. The researcher and node package
builds skip the React build because neither image serves the node GUI. Node.js,
Yarn, and the React compilation remain in the dedicated GUI image.

Every leg runs on an ephemeral machine, so the Docker build cache is discarded
with the runner and needs no cache management of its own.

The VPN build wrapper propagates the first failed Docker build instead of
continuing with later images. Cleanup removes resources created by the current
run and does not execute a broad `docker system prune`.

### Docker publication

`docker-deploy.yml` owns publication of the public base, node, and researcher
images. Published images use one Python runtime, currently Python 3.14. It is
the default `PYTHON_VERSION` in `docker/base/Dockerfile`; `docker-deploy.yml`
passes no build argument, so editing that default changes what is published.

A `v*.*.*` tag publishes the generated tags to Docker Hub; a manual run builds the
same images and publishes nothing. Build and push are the same job, so a failed
build publishes nothing. Nothing else triggers this workflow: `test-docker.yml`
builds the same three Dockerfiles every weekday across every supported interpreter, and
a release branch differs from `develop` only in files those Dockerfiles ignore,
so the release tree is already covered before a tag exists.

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
| `0 20 * * 1-5` | `test-docker.yml` | Monday to Friday at 20:00 UTC |
| `0 23 * * 1-5` | `end-to-end.yml` | Monday to Friday at 23:00 UTC |
| `0 9 * * 6` | `endurance-tests.yml` | Saturday at 09:00 UTC |
| `0 18 * * 1-5` | `build-test.yml` | Monday to Friday at 18:00 UTC |
| `0 1 * * 1` | `runner-maintenance.yml` | Monday at 01:00 UTC |
| `0 3 * * 1` | `package-compatibility.yml` | Monday at 03:00 UTC |

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

The Apple Silicon runner labelled `macos-m1` is the only self-hosted machine in
the matrices. It must provide the tools required by its jobs:

- a supported shell and Git
- passwordless access for the package-manager commands used by the setup action
- enough disk space for tox environments and test artifacts
- Homebrew, which the setup action uses to install interpreters

Runner labels are part of the workflow interface. If a runner is renamed or
relabelled, update every matrix that refers to its old label.

GitHub cannot address several self-hosted runners in one job, so
`runner-maintenance.yml` names its target. An additional runner is cleaned only
once a job for it is added to that workflow. A runner that is offline when the
workflow starts keeps its job queued until GitHub cancels it, so a failing job
means the machine needs attention.
