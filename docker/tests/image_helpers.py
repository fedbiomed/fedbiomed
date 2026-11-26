"""Helpers shared by the Docker image tests (`docker/tests/test_*.py`).

Every resource a test creates (container, network, volume) must carry `LABEL`
and have a name built with `unique()`: the `cleanup` fixture (conftest.py) and
the CI job remove resources by this label, and container logs are printed
when a test fails.
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

# Images under test, set by CI to the freshly built tags.
IMAGES = {
    "base": os.environ.get("FBM_BASE_IMAGE", "fedbiomed/base:latest"),
    "node": os.environ.get("FBM_NODE_IMAGE", "fedbiomed/node:latest"),
    "researcher": os.environ.get("FBM_RESEARCHER_IMAGE", "fedbiomed/researcher:latest"),
}
# Seconds to wait for a service in a container (component creation included).
TIMEOUT = float(os.environ.get("FBM_TEST_TIMEOUT", "180"))

LABEL = "fbm-docker-test=1"
RUN_ID = uuid.uuid4().hex[:8]

# Run with an image's python to probe a URL from inside a container:
# exit code 0 if an HTTP response is received (any status), 1 otherwise.
PROBE = (
    "import sys, urllib.request, urllib.error\n"
    "try:\n"
    "    urllib.request.urlopen(sys.argv[1], timeout=5)\n"
    "except urllib.error.HTTPError:\n"
    "    pass\n"
    "except Exception as e:\n"
    "    print(type(e).__name__, e)\n"
    "    sys.exit(1)\n"
)


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Runs a docker CLI command and returns the completed process."""
    proc = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=600
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"docker {' '.join(args)} failed ({proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return proc


def unique(name: str) -> str:
    """Resource name unique to this test run."""
    return f"fbm-docker-test-{RUN_ID}-{name}"


def run_container(
    image: str,
    name: str,
    *,
    env: dict | None = None,
    network: str | None = None,
    publish: tuple = (),
    extra: tuple = (),
    command: tuple = (),
) -> str:
    """Starts a labelled container in the background and returns its name.

    `publish` lists container ports to publish on a random 127.0.0.1 host port
    (read it back with `host_port`)."""
    args = ["run", "-d", "--name", name, "--label", LABEL]
    for key, value in (env or {}).items():
        args += ["-e", f"{key}={value}"]
    if network:
        args += ["--network", network]
    for port in publish:
        args += ["-p", f"127.0.0.1::{port}"]
    docker(*args, *extra, image, *command)
    return name


def logs(name: str) -> str:
    proc = docker("logs", name, check=False)
    return proc.stdout + proc.stderr


def is_running(name: str) -> bool:
    state = docker("inspect", "-f", "{{.State.Running}}", name, check=False)
    return state.stdout.strip() == "true"


def container_ips(name: str) -> dict:
    """Returns {network name: IPv4 address} of a container."""
    out = docker("inspect", "-f", "{{json .NetworkSettings.Networks}}", name).stdout
    return {net: cfg["IPAddress"] for net, cfg in json.loads(out).items()}


def host_port(name: str, port: int) -> int:
    """Returns the host port published for a container port."""
    out = docker("port", name, f"{port}/tcp").stdout.strip().splitlines()[0]
    return int(out.rsplit(":", 1)[1])


def listening_addresses(name: str, port: int) -> set:
    """IPv4 addresses with a TCP socket listening on `port`, read from
    /proc/net/tcp in the container's network namespace (no tools needed)."""
    out = docker("exec", name, "cat", "/proc/net/tcp", check=False).stdout
    addresses = set()
    for line in out.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 4 or fields[3] != "0A":  # 0A = LISTEN
            continue
        ip_hex, port_hex = fields[1].split(":")
        if int(port_hex, 16) == port:
            addresses.add(socket.inet_ntoa(bytes.fromhex(ip_hex)[::-1]))
    return addresses


def wait_listening(name: str, port: int, timeout: float = TIMEOUT) -> set:
    """Waits until something listens on `port` in the container.

    Returns the listening addresses; raises AssertionError (with the container
    logs) if the container stops or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_running(name):
            raise AssertionError(f"container {name} stopped:\n{logs(name)}")
        addresses = listening_addresses(name, port)
        if addresses:
            return addresses
        time.sleep(2)
    raise AssertionError(
        f"nothing listening on port {port} within {timeout}s in {name}:\n{logs(name)}"
    )


def http_ok_from_host(port: int, path: str = "/", retries: int = 15) -> bool:
    """True if 127.0.0.1:<port><path> on the host answers HTTP 200.

    Retries: a socket may listen slightly before the server handles requests."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=5
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, ConnectionError, OSError):
            if attempt < retries - 1:
                time.sleep(2)
    return False


def http_reachable_from_peer(image: str, network: str, url: str) -> bool:
    """True if a fresh container of `image` on `network` gets an HTTP answer."""
    proc = docker(
        "run", "--rm", "--label", LABEL, "--network", network,
        "--entrypoint", "python3", image, "-c", PROBE, url,
        check=False,
    )
    return proc.returncode == 0


def http_reachable_inside(name: str, url: str) -> bool:
    """True if `url` answers from inside container `name`."""
    proc = docker("exec", name, "python3", "-c", PROBE, url, check=False)
    return proc.returncode == 0


def remove_containers(names, print_logs: bool = False) -> None:
    for name in names:
        if print_logs:
            print(f"\n----- docker logs {name} -----\n{logs(name)}")
        docker("rm", "-f", "-v", name, check=False)


def run_containers() -> list:
    """Names of the containers created by this test run."""
    return docker(
        "ps", "-a", "--filter", f"label={LABEL}", "--filter", f"name={RUN_ID}",
        "--format", "{{.Names}}", check=False,
    ).stdout.split()
