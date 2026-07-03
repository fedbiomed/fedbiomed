import argparse
import ipaddress as ip
import os
import re
import subprocess
from string import Template
from typing import Any, Callable

import requests
import tabulate

#
# Script for handling wireguard VPN peer configurations via wg-easy REST API
# - launched in `vpnserver` container (Linux), thus can use some os-specific commands
#

# paths for config files
template_file = os.path.join(os.sep, "vpn", "config_templates", "config_%s.env")
peer_config_folder = os.path.join(os.sep, "config", "config_peers")
config_file = "config.env"
peer_types = ["researcher", "node", "management"]

# wg-easy API
WG_EASY_URL = os.environ.get("WG_EASY_URL", "http://localhost:51821")
WG_EASY_PASSWORD = os.environ.get("WG_EASY_PASSWORD", "")

# UID and GID to use when dropping privileges
init_uid = os.geteuid()
if "CONTAINER_UID" in os.environ:
    try:
        container_uid = int(os.environ["CONTAINER_UID"])
    except (TypeError, ValueError) as e:
        print(
            f"CRITICAL: bad type or value of CONTAINER_UID={os.environ['CONTAINER_UID']} : {e}"
        )
        exit(1)
else:
    container_uid = init_uid
init_gid = os.getegid()
if "CONTAINER_GID" in os.environ:
    try:
        container_gid = int(os.environ["CONTAINER_GID"])
    except (TypeError, ValueError) as e:
        print(
            f"CRITICAL: bad type or value of CONTAINER_GID={os.environ['CONTAINER_GID']} : {e}"
        )
        exit(1)
else:
    container_gid = init_gid


class WgEasyClient:
    """HTTP client for the wg-easy REST API."""

    def __init__(self, base_url: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._password = password
        self._session = requests.Session()
        self._authenticated = False

    def _authenticate(self) -> None:
        try:
            resp = self._session.post(
                f"{self._base_url}/api/session",
                json={"password": self._password},
                timeout=10,
            )
            resp.raise_for_status()
            self._authenticated = True
        except requests.RequestException as e:
            print(f"CRITICAL: wg-easy authentication failed : {e}")
            exit(1)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        if not self._authenticated:
            self._authenticate()
        try:
            resp = self._session.request(
                method, f"{self._base_url}{path}", timeout=10, **kwargs
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"CRITICAL: wg-easy API request {method} {path} failed : {e}")
            exit(1)

    def create_client(self, name: str) -> dict:
        return self._request(
            "POST", "/api/wireguard/client", json={"name": name}
        ).json()

    def delete_client(self, client_id: str) -> None:
        self._request("DELETE", f"/api/wireguard/client/{client_id}")

    def list_clients(self) -> list:
        return self._request("GET", "/api/wireguard/client").json()

    def get_client_config(self, client_id: str) -> str:
        return self._request(
            "GET", f"/api/wireguard/client/{client_id}/configuration"
        ).text


def run_drop_priv(function: Callable, *args, **kwargs) -> Any:
    """Run `function(args, kwargs)` with privileges of
    `container_uid:container_gid` (temporary drop)

    Args:
        - function (Callable) : the function to run
        - args : positional arguments to pass to the function
        - kwargs : named arguments to pass to the function

    Raises:

    Returns:
        - Any : return value of `function`
    """
    try:
        os.setegid(container_gid)
        os.seteuid(container_uid)
    except PermissionError as e:
        print(
            f"CRITICAL: cannot set identity to {container_uid}:{container_gid} : {e}"
            f"for running {function}"
        )
        exit(1)

    try:
        ret = function(*args, **kwargs)
    except Exception as e:
        print(
            f"CRITICAL: error while running function {function} with identity "
            f"{container_uid}:{container_gid} : {e}"
        )
        exit(1)

    try:
        os.seteuid(init_uid)
        os.setegid(init_gid)
    except PermissionError as e:
        print(
            f"CRITICAL: cannot restore identity to {container_uid}:{container_gid} : {e}"
            f"after running {function}"
        )
        exit(1)

    return ret


def read_config_file(filepath: str) -> dict:
    """Read a peer config.env file and build a dict from its content

    Args:
        - filepath (str) : path to the config file to read

    Raises:

    Returns:
        - dict : config file content as a dict of variable name (item key) and
            variable value (item value)
    """

    try:
        f = open(filepath, "r")
    except Exception as e:
        print(f"CRITICAL: cannot open config file {filepath} : {e}")
        exit(1)

    peer_config = {}
    for line in f:
        if not line.startswith("#") and not line == "" and not line.isspace():
            t = tuple(line.strip(" \n").removeprefix("export").lstrip().split("=", 1))
            if (
                not len(t) == 2
                or not isinstance(t[0], str)
                or not isinstance(t[1], str)
            ):
                print(f"CRITICAL: bad variable in config file {filepath} : {t}")
                exit(1)
            peer_config[t[0]] = t[1]
    f.close()

    for variable in peer_config.items():
        if (
            not len(variable) == 2
            or not isinstance(variable[0], str)
            or not isinstance(variable[1], str)
        ):
            print(f"CRITICAL: bad variable in config file {filepath} : {variable}")
            exit(1)

    return peer_config


def save_config_file(peer_type: str, peer_id: str, mapping: dict) -> None:
    """Save a new peer config.env file from a mapping dict

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer
        - mapping (dict) : dictionaries with the items needed to fill the peer
            configuration template

    Raises:

    Returns:
        - None
    """

    outpath = os.path.join(peer_config_folder, peer_type)
    run_drop_priv(os.makedirs, outpath, exist_ok=True)

    outpath = os.path.join(outpath, peer_id)
    run_drop_priv(os.mkdir, outpath)

    filepath = os.path.join(outpath, config_file)
    f_config = run_drop_priv(open, filepath, "w")

    try:
        f_template = open(template_file % peer_type, "r")
    except Exception as e:
        print(f"CRITICAL: cannot open template file {template_file % peer_type} : {e}")
        exit(1)

    f_config.write(Template(f_template.read()).substitute(mapping))
    f_config.close()

    print(f"info: configuration for {peer_type}/{peer_id} saved in {filepath}")


def remove_config_file(peer_type: str, peer_id: str) -> None:
    """Remove configuration file and directory for `peer_id`

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer

    Raises:

    Returns:
        - None
    """

    conf_dir = os.path.join(peer_config_folder, peer_type, peer_id)
    conf_file = os.path.join(conf_dir, config_file)
    if os.path.isdir(conf_dir) and os.path.isfile(conf_file):
        run_drop_priv(os.remove, conf_file)
        run_drop_priv(os.rmdir, conf_dir)
        print(f"info: removed config file {conf_file}")
    else:
        print(f"CRITICAL: missing configuration file {conf_file}")
        exit(1)


def parse_private_key_from_wg_config(wg_config_text: str) -> str:
    """Extract the PrivateKey value from a WireGuard config file text

    Args:
        - wg_config_text (str) : raw text of a WireGuard client config file

    Raises:

    Returns:
        - str : the base64-encoded private key
    """
    for line in wg_config_text.splitlines():
        m = re.match(r"^\s*PrivateKey\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    print("CRITICAL: could not parse PrivateKey from wg-easy client configuration")
    exit(1)


def get_server_public_key() -> str:
    """Get the WireGuard server's public key via wg CLI

    Args:

    Raises:

    Returns:
        - str : base64-encoded server public key
    """
    try:
        return subprocess.run(
            ["wg", "show", "wg0", "public-key"],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
        ).stdout.rstrip("\n")
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: server public key retrieve failed with error : {e}")
        exit(1)


def check_peer_args(peer_type: str, peer_id: str) -> None:
    """Check peer arguments type and value

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer

    Raises:

    Returns:
        - None
    """

    if not isinstance(peer_type, str):
        print(f"CRITICAL: bad type for `peer_type` {peer_type}")
        exit(1)
    if not isinstance(peer_id, str):
        print(f"CRITICAL: bad type for `peer_id` {peer_id}")
        exit(1)
    if peer_type not in peer_types:
        print(f"CRITICAL: bad value for `peer_type` {peer_type}")
        exit(1)


def _require_wg_easy_password() -> None:
    if not WG_EASY_PASSWORD:
        print("CRITICAL: WG_EASY_PASSWORD environment variable is not set")
        exit(1)


def genconf(peer_type: str, peer_id: str) -> None:
    """Generate and save configuration for a new peer via wg-easy API.

    Creates the peer in wg-easy (which assigns an IP and generates a keypair),
    then saves a config.env for the peer containing all connection parameters
    including the wg-easy-generated private key.

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer

    Raises:

    Returns:
        - None
    """

    check_peer_args(peer_type, peer_id)
    _require_wg_easy_password()

    required_env = {
        "VPN_SUBNET_PREFIX",
        "VPN_SERVER_PUBLIC_ADDR",
        "VPN_SERVER_PORT",
        "VPN_IP",
    }
    if not required_env <= os.environ.keys():
        print(
            "CRITICAL: missing environment variables : "
            "`VPN_SUBNET_PREFIX` or `VPN_SERVER_PUBLIC_ADDR` or `VPN_SERVER_PORT` or `VPN_IP`"
        )
        exit(1)

    # don't update if config for this peer already exists
    if os.path.isfile(
        os.path.join(peer_config_folder, peer_type, peer_id, config_file)
    ):
        print(
            f"WARNING: do nothing, config file already exists for peer {peer_type}/{peer_id}"
        )
        return

    # create client in wg-easy: assigns IP, generates keypair and PSK, activates peer
    wg_easy = WgEasyClient(WG_EASY_URL, WG_EASY_PASSWORD)
    client = wg_easy.create_client(f"{peer_type}/{peer_id}")

    client_id = client.get("id")
    peer_addr_str = client.get("address")
    peer_psk = client.get("preSharedKey")

    if not client_id or not peer_addr_str or not peer_psk:
        print(f"CRITICAL: incomplete response from wg-easy client creation : {client}")
        exit(1)

    # fetch client WireGuard config to extract the private key
    client_wg_config = wg_easy.get_client_config(client_id)
    peer_private_key = parse_private_key_from_wg_config(client_wg_config)

    # validate the assigned IP is within the VPN network
    try:
        peer_addr_ip = ip.IPv4Address(peer_addr_str)
        vpn_net = ip.IPv4Interface(
            f"{os.environ['VPN_IP']}/{os.environ['VPN_SUBNET_PREFIX']}"
        ).network
    except (ip.AddressValueError, ip.NetmaskValueError) as e:
        print(f"CRITICAL: IP address validation failed : {e}")
        exit(1)

    if peer_addr_ip not in vpn_net:
        print(
            f"CRITICAL: assigned peer IP {peer_addr_ip} is not in VPN network {vpn_net}"
        )
        exit(1)

    server_public_key = get_server_public_key()

    mapping = {
        "vpn_ip": peer_addr_ip,
        "vpn_subnet_prefix": os.environ["VPN_SUBNET_PREFIX"],
        "vpn_server_endpoint": f"{os.environ['VPN_SERVER_PUBLIC_ADDR']}:{os.environ['VPN_SERVER_PORT']}",
        "vpn_server_allowed_ips": str(vpn_net),
        "vpn_server_public_key": server_public_key,
        "vpn_server_psk": peer_psk,
        "vpn_private_key": peer_private_key,
        "fedbiomed_id": peer_id,
        "fedbiomed_net_ip": os.environ["VPN_IP"],
    }

    if None in mapping.values() or "" in mapping.values():
        print(f"CRITICAL: bad value in configuration mapping {mapping}")
        exit(1)

    save_config_file(peer_type, peer_id, mapping)


def add(peer_type: str, peer_id: str, peer_public_key: str) -> None:
    """No-op: peer registration now happens at genconf time via wg-easy.

    This subcommand is kept for backward compatibility. In the wg-easy flow the
    peer is created and immediately activated when `genconf` is run, so there is
    no separate step to register the peer's public key.

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer
        - peer_public_key (str) : public key of the peer (ignored)

    Raises:

    Returns:
        - None
    """

    check_peer_args(peer_type, peer_id)
    print(
        f"info: `add` is no longer required — peer {peer_type}/{peer_id} was registered "
        f"in wg-easy when `genconf` was run. No action taken."
    )


def remove(peer_type: str, peer_id: str, removeconf: bool = False) -> None:
    """Remove a peer from wg-easy and optionally remove its local config file

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer
        - removeconf (bool, optional) : whether to also remove the config file.
            Defaults to False.

    Raises:

    Returns:
        - None
    """

    check_peer_args(peer_type, peer_id)
    if not isinstance(removeconf, bool):
        print(f"CRITICAL: bad type for `removeconf` {removeconf}")
        exit(1)

    _require_wg_easy_password()

    filepath = os.path.join(peer_config_folder, peer_type, peer_id, config_file)
    if not os.path.isfile(filepath):
        print(
            f"ERROR: do nothing, peer {peer_type}/{peer_id} does not exist. "
            "You need to create it with `genconf` first."
        )
        return

    peer_config = read_config_file(filepath)
    if "VPN_IP" not in peer_config:
        print("CRITICAL: missing entry in peer config file : `VPN_IP`")
        exit(1)

    # find the wg-easy client by IP address and delete it
    wg_easy = WgEasyClient(WG_EASY_URL, WG_EASY_PASSWORD)
    clients = wg_easy.list_clients()

    client_id = None
    for c in clients:
        if c.get("address") == peer_config["VPN_IP"]:
            client_id = c.get("id")
            break

    if client_id is None:
        print(
            f"WARNING: no wg-easy client found with IP {peer_config['VPN_IP']}, "
            "may have been removed already"
        )
    else:
        wg_easy.delete_client(client_id)
        print(
            f"info: successfully removed wg-easy client for peer {peer_type}/{peer_id} "
            f"(IP {peer_config['VPN_IP']})"
        )

    if removeconf:
        remove_config_file(peer_type, peer_id)


def list_peers() -> None:
    """List peers from wg-easy API cross-referenced with local config files,
    pretty print on standard output

    Args:

    Raises:

    Returns:
        - None
    """

    _require_wg_easy_password()

    # scan local config files: IP prefix -> {type, id}
    peers = {}
    if os.path.isdir(peer_config_folder):
        for peer_type in os.listdir(peer_config_folder):
            pt_path = os.path.join(peer_config_folder, peer_type)
            if not os.path.isdir(pt_path):
                continue
            for peer_id in os.listdir(pt_path):
                filepath = os.path.join(pt_path, peer_id, config_file)
                if not os.path.isfile(filepath):
                    continue
                peer_config = read_config_file(filepath)
                try:
                    peer_ip = str(ip.IPv4Network(f"{peer_config['VPN_IP']}/32"))
                except (ip.AddressValueError, ip.NetmaskValueError, KeyError) as e:
                    print(f"CRITICAL: bad `VPN_IP` in {filepath} : {e}")
                    exit(1)
                peers[peer_ip] = {
                    "type": peer_type,
                    "id": peer_id,
                    "active": False,
                    "wg_easy_id": None,
                }

    # cross-reference with wg-easy clients
    wg_easy = WgEasyClient(WG_EASY_URL, WG_EASY_PASSWORD)
    for c in wg_easy.list_clients():
        addr = c.get("address", "")
        try:
            client_ip = str(ip.IPv4Network(f"{addr}/32")) if addr else None
        except (ip.AddressValueError, ip.NetmaskValueError):
            client_ip = None

        if client_ip and client_ip in peers:
            peers[client_ip]["active"] = c.get("enabled", False)
            peers[client_ip]["wg_easy_id"] = c.get("id", "")
        elif client_ip:
            peers[client_ip] = {
                "type": "?",
                "id": c.get("name", "?"),
                "active": c.get("enabled", False),
                "wg_easy_id": c.get("id", ""),
            }

    pretty_peers = [
        [v["type"], v["id"], k, v["active"], v["wg_easy_id"]] for k, v in peers.items()
    ]
    print(
        tabulate.tabulate(
            pretty_peers, headers=["type", "id", "prefix", "active", "wg_easy_id"]
        )
    )


#
# Main
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure Wireguard peers on the server via wg-easy API"
    )
    subparsers = parser.add_subparsers(
        dest="operation", required=True, help="operation to perform"
    )

    parser_genconf = subparsers.add_parser(
        "genconf", help="generate the config file for a new peer"
    )
    parser_genconf.add_argument(
        "type", choices=peer_types, help="type of client to generate config for"
    )
    parser_genconf.add_argument("id", type=str, help="id of the new peer")

    parser_add = subparsers.add_parser(
        "add", help="(deprecated) no-op: peer is registered at genconf time via wg-easy"
    )
    parser_add.add_argument("type", choices=peer_types, help="type of client to add")
    parser_add.add_argument("id", type=str, help="id of the client")
    parser_add.add_argument(
        "publickey", type=str, help="publickey of the client (ignored)"
    )

    parser_remove = subparsers.add_parser("remove", help="remove a peer from wg-easy")
    parser_remove.add_argument(
        "type", choices=peer_types, help="type of client to remove"
    )
    parser_remove.add_argument("id", type=str, help="id of client to remove")

    parser_removeconf = subparsers.add_parser(
        "removeconf", help="remove a peer from wg-easy and delete its config file"
    )
    parser_removeconf.add_argument(
        "type", choices=peer_types, help="type of client to remove"
    )
    parser_removeconf.add_argument("id", type=str, help="id of client to remove")

    parser_list = subparsers.add_parser(
        "list", help="list peers and their wg-easy status"
    )

    args = parser.parse_args()

    if args.operation == "genconf":
        genconf(args.type, args.id)
    elif args.operation == "add":
        add(args.type, args.id, args.publickey)
    elif args.operation == "remove":
        remove(args.type, args.id)
    elif args.operation == "removeconf":
        remove(args.type, args.id, True)
    elif args.operation == "list":
        list_peers()
