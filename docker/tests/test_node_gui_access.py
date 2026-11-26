"""End-to-end tests of the Node GUI listening address in the `fedbiomed/node` image.

The node image starts the GUI with supervisord, bound to `FBM_GUI_HOST`, which
`docker/node/entrypoint.sh` resolves at container start:

- unset: the container's own Docker-network IP (its hostname in /etc/hosts),
  falling back to 127.0.0.1 when it cannot be resolved;
- otherwise: the given address (e.g. 127.0.0.1 or 0.0.0.0).

A second Docker network attached to a "sidecar" container stands in for a VPN
interface: the node shares the sidecar's network (`--network container:...`),
like `network_mode: service:wg-client` in the VPN deployment guide
(GHSA-86g6-36qm-vx82).
"""

import pytest

from image_helpers import (
    container_ips,
    docker,
    host_port,
    http_ok_from_host,
    http_reachable_from_peer,
    http_reachable_inside,
    logs,
    run_container,
    unique,
    wait_listening,
)

GUI_PORT = 8484
FIXED_SUBNET = "10.199.77.0/24"
FIXED_IP = "10.199.77.10"


def gui_url(ip: str) -> str:
    return f"http://{ip}:{GUI_PORT}/"


def run_node(image: str, name: str, *, publish: bool = True, **kwargs) -> str:
    """Starts a node container (MNIST download disabled)."""
    env = {"MNIST_OFF": "True", **kwargs.pop("env", {})}
    return run_container(
        image, unique(name), env=env,
        publish=(GUI_PORT,) if publish else (), **kwargs,
    )


def wait_gui_listening(name: str) -> str:
    """Waits until the GUI listens and returns its (single) bound address."""
    addresses = wait_listening(name, GUI_PORT)
    assert len(addresses) == 1, f"GUI listens on several addresses: {addresses}"
    return addresses.pop()


def rendered_command(name: str) -> str:
    """The GUI command line rendered in the supervisord config, plus a space."""
    out = docker(
        "exec", name, "grep", "^command=", "/etc/supervisor/supervisord.conf"
    ).stdout
    return out.strip() + " "


@pytest.fixture
def sidecar(node_image, network):
    """A container attached to `pub` (GUI port published) then `vpn`, whose
    network the node shares, like a WireGuard client sidecar.

    Returns (sidecar name, pub network, vpn network)."""
    pub, vpn = network("pub"), network("vpn")
    name = run_container(
        node_image, unique("sidecar"), network=pub, publish=(GUI_PORT,),
        extra=("--entrypoint", "sleep"), command=("infinity",),
    )
    docker("network", "connect", vpn, name)
    return name, pub, vpn


def test_default_binds_container_ip_and_host_can_access(node_image):
    """FBM_GUI_HOST unset: GUI on the container IP, reachable through -p."""
    name = run_node(node_image, "default")
    bound = wait_gui_listening(name)

    ips = list(container_ips(name).values())
    assert bound in ips, f"GUI bound to {bound}, container IPs are {ips}"
    assert f"--host {bound} " in rendered_command(name)
    assert f"[INFO] Node GUI listening on {bound}:{GUI_PORT}" in logs(name)
    assert http_ok_from_host(host_port(name, GUI_PORT)), "GUI not reachable from host"


def test_default_not_reachable_on_second_interface(node_image, sidecar):
    """FBM_GUI_HOST unset, node sharing a sidecar network with a second
    ("VPN") interface: host access works, the second interface does not
    serve the GUI (GHSA-86g6-36qm-vx82)."""
    side, pub, vpn = sidecar
    name = run_node(
        node_image, "default-vpn", network=f"container:{side}", publish=False
    )
    bound = wait_gui_listening(name)

    ips = container_ips(side)
    # Behaviour first, so that a regression reports the exposure itself.
    assert not http_reachable_from_peer(node_image, vpn, gui_url(ips[vpn])), (
        "GUI reachable on the second (VPN) interface"
    )
    assert http_ok_from_host(host_port(side, GUI_PORT)), "GUI not reachable from host"
    assert bound == ips[pub], f"GUI bound to {bound}, expected {ips[pub]}"


def test_all_interfaces_0000(node_image, sidecar):
    """FBM_GUI_HOST=0.0.0.0: GUI on all interfaces, including the second one,
    with a warning in the logs."""
    side, _, vpn = sidecar
    name = run_node(
        node_image, "all", env={"FBM_GUI_HOST": "0.0.0.0"},
        network=f"container:{side}", publish=False,
    )
    assert wait_gui_listening(name) == "0.0.0.0"
    assert "--host 0.0.0.0 " in rendered_command(name)
    assert "[WARNING] Node GUI exposed on all interfaces" in logs(name)

    assert http_ok_from_host(host_port(side, GUI_PORT)), "GUI not reachable from host"
    assert http_reachable_from_peer(
        node_image, vpn, gui_url(container_ips(side)[vpn])
    ), "GUI not reachable on the second interface with 0.0.0.0"


def test_loopback_only_127001(node_image, network):
    """FBM_GUI_HOST=127.0.0.1: GUI reachable from inside the container only."""
    pub = network("pub")
    name = run_node(
        node_image, "loopback", env={"FBM_GUI_HOST": "127.0.0.1"}, network=pub
    )
    assert wait_gui_listening(name) == "127.0.0.1"
    assert http_reachable_inside(name, gui_url("127.0.0.1")), "GUI not reachable inside"

    assert not http_ok_from_host(host_port(name, GUI_PORT), retries=2), (
        "GUI bound to 127.0.0.1 is reachable from the host"
    )
    assert not http_reachable_from_peer(
        node_image, pub, gui_url(container_ips(name)[pub])
    ), "GUI bound to 127.0.0.1 is reachable from another container"


def test_explicit_ip_from_env(node_image, network):
    """FBM_GUI_HOST=<ip>: GUI bound to exactly that address."""
    fixed = network("fixed", "--subnet", FIXED_SUBNET)
    name = run_node(
        node_image, "explicit", env={"FBM_GUI_HOST": FIXED_IP},
        network=fixed, extra=("--ip", FIXED_IP),
    )
    assert wait_gui_listening(name) == FIXED_IP
    assert f"--host {FIXED_IP} " in rendered_command(name)
    assert http_ok_from_host(host_port(name, GUI_PORT)), "GUI not reachable from host"


def test_env_change_requires_recreate(node_image, volume):
    """The address is resolved at every container start: recreating the
    container with another FBM_GUI_HOST (same /fbm-node volume) applies it."""
    mount = ("-v", f"{volume}:/fbm-node")
    first = run_node(
        node_image, "recreate-1", env={"FBM_GUI_HOST": "127.0.0.1"}, extra=mount
    )
    assert wait_gui_listening(first) == "127.0.0.1"
    docker("rm", "-f", first)

    second = run_node(node_image, "recreate-2", extra=mount)
    assert wait_gui_listening(second) in container_ips(second).values()
    assert http_ok_from_host(host_port(second, GUI_PORT)), "GUI not reachable from host"


def test_no_network_falls_back_to_loopback(node_image):
    """No container IP (--network none): fall back to 127.0.0.1 with a warning."""
    name = run_node(node_image, "none", network="none", publish=False)
    assert wait_gui_listening(name) == "127.0.0.1"
    assert "Could not resolve the container IP" in logs(name)
