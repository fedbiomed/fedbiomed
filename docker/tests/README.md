# Docker image tests

End-to-end tests of the Fed-BioMed Docker images (`fedbiomed/base`, `fedbiomed/node`, `fedbiomed/researcher`). They start real containers, so they need a Docker daemon and the images to test.

They are kept outside `tests/` so the unit test job, which runs `pytest` from `tests/`, does not collect them.

| File | Content |
| --- | --- |
| `conftest.py` | Shared fixtures: Docker check, images under test, cleanup, networks, volumes |
| `image_helpers.py` | Shared helpers: run containers, read logs and IPs, wait for a listening port, HTTP probes |
| `test_*.py` | Test modules, one per topic, e.g. `test_node_gui_access.py` |

## Running locally

Build the images (see the [developer documentation](../../docs/developer/docker-images.md)), then run:

```bash
python -m pip install pytest
FBM_NODE_IMAGE=fedbiomed/node:<tag> python -m pytest -v docker/tests
# a single module or test
python -m pytest -v docker/tests/test_node_gui_access.py -k loopback
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `FBM_BASE_IMAGE` | `fedbiomed/base:latest` | Base image under test |
| `FBM_NODE_IMAGE` | `fedbiomed/node:latest` | Node image under test |
| `FBM_RESEARCHER_IMAGE` | `fedbiomed/researcher:latest` | Researcher image under test |
| `FBM_TEST_TIMEOUT` | `180` | Seconds to wait for a service in a container |

The tests publish ports on random `127.0.0.1` host ports and only touch resources they create. They can run next to other Fed-BioMed containers.

## Adding a test

1. Add a `test_<topic>.py` module in this folder. The CI job runs every module here, so the workflow does not need to change.
2. Request the image you test as a fixture: `base_image`, `node_image` or `researcher_image`. A missing image fails the test with a clear message.
3. Create resources through the helpers and fixtures. They carry the `fbm-docker-test=1` label and a run-specific name, so they are removed after each test, and their logs are printed when a test fails:
   - containers: `image_helpers.run_container(image, unique("name"), ...)`
   - networks: the `network` fixture, `network("name", "--subnet", "...")`
   - volumes: the `volume` fixture
4. Wait for services with `wait_listening(container, port)` rather than fixed sleeps.

```python
from image_helpers import run_container, unique, wait_listening


def test_researcher_starts_jupyter(researcher_image):
    name = run_container(researcher_image, unique("researcher"))
    assert wait_listening(name, 8888)
```

## CI

The `Docker image tests` job of `.github/workflows/test-docker.yml` builds the three images from the branch and runs `pytest docker/tests`. It runs:

- on pull requests to `develop` or `master` that change `docker/**`, `fedbiomed/node/cli.py`, `fedbiomed_node_api/**` or the workflow;
- every weekday evening;
- on demand, on any branch:

```bash
gh workflow run test-docker.yml --ref <branch> -f suite=image-tests -f os='["ubuntu-latest"]'
# a subset of the tests, on several runners
gh workflow run test-docker.yml --ref <branch> -f suite=image-tests \
  -f os='["ubuntu-24.04","ubuntu-24.04-arm"]' -f pytest-args='-k gui'
```
