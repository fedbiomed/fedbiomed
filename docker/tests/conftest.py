"""pytest configuration and shared fixtures for the Docker image tests.

These tests drive real containers and need a Docker daemon. They live outside
`tests/` on purpose so that the unit test job (which runs `pytest` from
`tests/`) does not collect them. See README.md in this folder.
"""

import os
import sys

import pytest

# Make `image_helpers` importable from the test modules (pytest runs with
# --import-mode=importlib, which does not add test folders to sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_helpers import (  # noqa: E402
    IMAGES,
    LABEL,
    docker,
    remove_containers,
    run_containers,
    unique,
)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Expose each phase's report on the test item (`item.rep_setup`,
    `item.rep_call`, ...) so fixtures can print container logs on failure."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


def failed(request) -> bool:
    """True if the setup or call phase of the requesting test failed."""
    return any(
        getattr(request.node, f"rep_{when}", None) is not None
        and getattr(request.node, f"rep_{when}").failed
        for when in ("setup", "call")
    )


@pytest.fixture(scope="session", autouse=True)
def docker_available():
    """Stops the session early when there is no Docker daemon."""
    try:
        docker("info", "--format", "{{.ServerVersion}}")
    except (RuntimeError, FileNotFoundError) as e:
        pytest.exit(f"Docker daemon not available: {e}", returncode=2)


def _image(kind: str) -> str:
    image = IMAGES[kind]
    if docker("image", "inspect", image, check=False).returncode != 0:
        pytest.fail(
            f"image under test not found: {image} "
            f"(build it, or set FBM_{kind.upper()}_IMAGE)",
            pytrace=False,
        )
    return image


@pytest.fixture(scope="session")
def base_image() -> str:
    """Base image under test (FBM_BASE_IMAGE)."""
    return _image("base")


@pytest.fixture(scope="session")
def node_image() -> str:
    """Node image under test (FBM_NODE_IMAGE)."""
    return _image("node")


@pytest.fixture(scope="session")
def researcher_image() -> str:
    """Researcher image under test (FBM_RESEARCHER_IMAGE)."""
    return _image("researcher")


@pytest.fixture(autouse=True)
def cleanup(request):
    """Removes the containers of this run after each test; prints their logs
    if the test failed."""
    yield
    remove_containers(run_containers(), failed(request))


@pytest.fixture
def volume(request):
    """A labelled named volume, removed after the test."""
    name = unique("vol")
    docker("volume", "create", "--label", LABEL, name)
    yield name
    # Runs before `cleanup`: remove the containers using the volume first.
    users = docker(
        "ps", "-a", "--filter", f"volume={name}", "--format", "{{.Names}}",
        check=False,
    ).stdout.split()
    remove_containers(users, failed(request))
    docker("volume", "rm", "-f", name, check=False)


@pytest.fixture
def network(request):
    """Factory creating labelled networks, removed after the test:
    `network("name", "--subnet", "10.1.2.0/24")`."""
    created = []

    def create(name: str, *options: str) -> str:
        full = unique(name)
        docker("network", "create", "--label", LABEL, *options, full)
        created.append(full)
        return full

    yield create
    # Runs before `cleanup`: remove the containers attached to the networks.
    if created:
        remove_containers(run_containers(), failed(request))
    for net in created:
        docker("network", "rm", net, check=False)
