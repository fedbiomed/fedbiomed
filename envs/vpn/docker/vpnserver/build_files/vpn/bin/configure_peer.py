import argparse
from string import Template
import subprocess
import os
import re
import ipaddress as ip
from typing import Any, Callable

import tabulate

#
# Script for handling wireguard VPN peer configurations
# - launched in `vpnserver` container (Linux), thus can use some os-specific commands
#

#
# Initialize variables
#

# paths templates for config files
# TODO: name global variables in upper case
template_file = os.path.join(os.sep, 'fedbiomed', 'vpn', 'config_templates', 'config_%s.env')
assign_config_file = os.path.join(os.sep, 'config', 'ip_assign', 'last_ip_assign_%s')
peer_config_folder = os.path.join(os.sep, 'config', 'config_peers')
wg_config_file = os.path.join(os.sep, 'config', 'wireguard', 'wg0.conf')
config_file = 'config.env'
peer_types = ["researcher", "node", "management"]  # use tuple instead?

# UID and GID to use when dropping privileges
init_uid = os.geteuid()
if 'CONTAINER_UID' in os.environ:
    try:
        container_uid = int(os.environ['CONTAINER_UID'])
    except (TypeError, ValueError) as e:
        print(f"CRITICAL: bad type or value of CONTAINER_UID={os.environ['CONTAINER_UID']} : {e}")
        exit(1)
else:
    container_uid = init_uid
init_gid = os.getegid()
if 'CONTAINER_GID' in os.environ:
    try:
        container_gid = int(os.environ['CONTAINER_GID'])
    except (TypeError, ValueError) as e:
        print(f"CRITICAL: bad type or value of CONTAINER_GID={os.environ['CONTAINER_GID']} : {e}")
        exit(1)    
else:
    container_gid = init_gid


#
# Functions
#


def run_drop_priv(function: Callable, *args, **kwargs) -> Any:
    """ Run `function(args, kwargs)` with privileges of
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
        print(f"CRITICAL: cannot set identity to {container_uid}:{container_gid} : {e}"
              f"for running {function}")
        exit(1)

    try:
        ret = function(*args, **kwargs)
    except Exception as e:
        print(f"CRITICAL: error while running function {function} with identity "
              f"{container_uid}:{container_gid} : {e}")
        exit(1)        

    try:
        os.seteuid(init_uid)
        os.setegid(init_gid)
    except PermissionError as e:
        print(f"CRITICAL: cannot restore identity to {container_uid}:{container_gid} : {e}"
              f"after running {function}")
        exit(1)

    return ret


def read_config_file(filepath: str) -> dict:
    """ Read a peer config.env file and build a dict from its content

    Args:
        - filepatch (str) : path to the config file to read

    Raises:

    Returns:
        - dict : config file content as a dict of variable name (item key) and
            variable value (item value)
    """

    try:    
        f = open(filepath, 'r')
    except Exception as e:
        print(f"CRITICAL: cannot open config file {filepath} : {e}")
        exit(1)

    peer_config = {}
    for line in f:
        if not line.startswith('#') and not line == '' and not line.isspace():
            t = tuple(line.strip(" \n").removeprefix('export').lstrip().split('=', 1))
            if not len(t) == 2 or not isinstance(t[0], str) or \
                    not isinstance(t[1], str):
                print(f"CRITICAL: bad variable in config file {filepath} : {t}")
                exit(1)
            peer_config[t[0]] = t[1]      
    f.close()

    for variable in peer_config.items():
        if not len(variable) == 2 or not isinstance(variable[0], str) or \
                not isinstance(variable[1], str):
            print(f"CRITICAL: bad variable in config file {filepath} : {variable}")
            exit(1)

    return peer_config


def save_config_file(peer_type: str, peer_id: str, mapping: dict) -> None:
    """ Save a new peer config.env file from a mapping dict

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
    f_config = run_drop_priv(open, filepath, 'w')

    try:
        f_template = open(template_file % peer_type, 'r')
    except Exception as e:
        print(f"CRITICAL: cannot open template file {f_template} % {peer_type} : {e}")
        exit(1)

    f_config.write(Template(f_template.read()).substitute(mapping))
    f_config.close()

    print(f"info: configuration for {peer_type}/{peer_id} saved in {filepath}")


def remove_config_file(peer_type: str, peer_id: str) -> None:
    """ Remove configuration file and directory for `peer_id`

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


def save_wg_file() -> None:
    """ Save wireguard config file from current wireguard interface params

    Args:

    Raises:

    Returns:
        - None
    """

    # read current wireguard interface config
    try:
        wg_config = subprocess.run(
            ['wg', 'showconf', 'wg0'],
            stdout=subprocess.PIPE,
            check=True).stdout.decode().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: wireguard config file read failed with error : {e}")
        exit(1)

    # save wireguard config to file
    f = run_drop_priv(open, wg_config_file, 'w')
    for line in wg_config:
        f.write(f'{line}\n')    # need to insert proper line breaks
    f.close()


def get_current_peers() -> list[list]:
    """ List peers currently declared in the current wireguard interface params

    Args:

    Raises:

    Returns:
        - list[list] : list of peers, each element is a 2-strings list. First item
            is the peer's public key, second item is the peer's IP prefix
    """

    current_peers = []

    try:
        f = subprocess.run(
            ['wg', 'show', 'wg0', 'allowed-ips'],
            stdout=subprocess.PIPE,
            check=True
        ).stdout.decode().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: wireguard current config read failed with error : {e}")
        exit(1)

    for line in f:
        peer = re.split('\\s+', line.strip(" \n"))
        if not len(peer) == 2 and isinstance(peer[0], str) and isinstance(peer[1], str):
            print(f"CRITICAL: wireguard current config gives incorrect output line : {peer}")
            exit(1)            
        current_peers.append(peer)

    return current_peers


# check peer arguments type and value
def check_peer_args(peer_type: str, peer_id: str) -> None:
    """ Check peer arguments type and value

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


def genconf(peer_type: str, peer_id: str) -> None:
    """ Generate and save configuration for a new peer
    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer

    Raises:

    Returns:
        - None
    """

    # check arguments
    check_peer_args(peer_type, peer_id)

    if not { 'VPN_SUBNET_PREFIX', 'VPN_SERVER_PUBLIC_ADDR', 'VPN_SERVER_PORT', 'VPN_IP',
             f"VPN_{peer_type.upper()}_IP_ASSIGN" } <= os.environ.keys():
        print("CRITICAL: missing environment variables in environment : "
              "`VPN_SUBNET_PREFIX` or `VPN_SERVER_PUBLIC_ADDR` or `VPN_SERVER_PORT` or `VPN_IP`")
        exit(1)

    # don't update if config for this peer already exist => error
    if os.path.isfile(os.path.join(peer_config_folder, peer_type, peer_id, config_file)):
        print(f"WARNING: do nothing, config file already exists for peer {peer_type}/{peer_id}")
        return

    # assign IP address for the new peer + save updated counter of assigned IP
    try:
        assign_net = ip.IPv4Network(os.environ[f"VPN_{peer_type.upper()}_IP_ASSIGN"])
    except (ip.AddressValueError, ip.NetmaskValueError, KeyError) as e:
        print(f"CRITICAL: bad `VPN_{peer_type}_IP_ASSIGN` in config gives errors: {e}")
        exit(1)

    assign_file = assign_config_file % peer_type
    if os.path.exists(assign_file) and os.path.getsize(assign_file) > 0:
        f = run_drop_priv(open, assign_file, 'r+')

        # assign the next available ip to the peer
        try:
            peer_addr_ip = ip.IPv4Address(f.read()) + 1
        except (ip.AddressValueError, ip.NetmaskValueError) as e:
            print(f"CRITICAL: bad IP address in assign file {assign_file} gives errors: {e}")
            exit(1)

        f.seek(0)
    else:
        f = run_drop_priv(open, assign_file, 'w')

        peer_addr_ip = assign_net.network_address + 2

    f.write(str(peer_addr_ip))
    f.close()


    # create configuration for the new peer
    try:
        peer_psk = subprocess.run(
            ["wg", "genpsk"],
            stdout=subprocess.PIPE,
            check=True,
            text=True
        ).stdout.rstrip('\n')
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: peer PSK generation failed with error : {e}")
        exit(1)
    try:
        server_public_key = subprocess.run(
            ["wg", "show", "wg0", "public-key"],
            stdout=subprocess.PIPE,
            check=True,
            text=True
        ).stdout.rstrip('\n')
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: server public key retrieve failed with error : {e}")
        exit(1)    

    try:
        vpn_net = ip.IPv4Interface(f"{os.environ['VPN_IP']}/{os.environ['VPN_SUBNET_PREFIX']}").network
    except (ip.AddressValueError, ip.NetmaskValueError, KeyError) as e:
        print(f"CRITICAL: bad interface VPN_IP/VPN_SUBNET_PREFIX "
              f"{os.environ['VPN_IP']}/{os.environ['VPN_SUBNET_PREFIX']} gives error : {e}")
        exit(1)

    if peer_addr_ip not in assign_net or peer_addr_ip not in vpn_net:
        print(f"CRITICAL: assigned peer IP {peer_addr_ip} is "
              f"not in VPN network {vpn_net} or not in assign range {assign_net}")
        exit(1)

    mapping = {
        "vpn_ip": peer_addr_ip,
        "vpn_subnet_prefix": os.environ['VPN_SUBNET_PREFIX'],
        "vpn_server_endpoint": f"{os.environ['VPN_SERVER_PUBLIC_ADDR']}:{os.environ['VPN_SERVER_PORT']}",
        "vpn_server_allowed_ips": str(vpn_net),
        "vpn_server_public_key": server_public_key,
        "vpn_server_psk": peer_psk,
        "fedbiomed_id": peer_id,
        "fedbiomed_net_ip": os.environ['VPN_IP']
    }

    if None in mapping.values() or "" in mapping.values():
        print(f"CRITICAL: bad value in configuration mapping {mapping}")
        exit(1)

    # save new peer configuration file
    save_config_file(peer_type, peer_id, mapping)


def add(peer_type: str, peer_id: str, peer_public_key: str) -> None:
    """ Finish definition of a new peer with peer's public key,
    in current wireguard interface and wireguard file

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer
        - peer_public_key (str) : public key of the peer

    Raises:

    Returns:
        - None
    """

    # check arguments
    check_peer_args(peer_type, peer_id)
    if not isinstance(peer_public_key, str):
        print(f"CRITICAL: bad type for `peer_public_key` {peer_public_key}")
        exit(1)

    # read peer config file
    filepath = os.path.join(peer_config_folder, peer_type, peer_id, config_file)
    if not os.path.isfile(filepath):
        # want explicit message and failure
        print(f"ERROR: do nothing, peer {peer_type}/{peer_id} does not exist. "
              "You need to create it with `genconf` first.")
        return

    peer_config = read_config_file(filepath)
    if not { 'VPN_SERVER_PSK', 'VPN_IP' } <= peer_config.keys():
        print("CRITICAL: missing entries in peer config file : `VPN_SERVER_PSK` or `VPN_IP`")
        exit(1)    

    # add the new peer to the current wireguard interface config
    try:
        subprocess.run(
            ["wg", "set", "wg0", "peer", peer_public_key, "allowed-ips",
                str(ip.IPv4Network(f"{peer_config['VPN_IP']}/32")),
                "preshared-key", "/dev/stdin"],
            input=peer_config['VPN_SERVER_PSK'],
            check=True,
            text=True
        )
    except (subprocess.CalledProcessError, ip.AddressValueError, ip.NetmaskValueError)as e:
        print(f"CRITICAL: setting peer in wireguard interface failed with error : {e}")
        exit(1)

    # save updated wireguard config file
    save_wg_file()


def remove(peer_type: str, peer_id: str, removeconf: bool = False) -> None:
    """ Remove a peer from the current wireguard interface configuration,
    save updated wireguard config file, and optionally remove peer config file

    Args:
        - peer_type (str) : whether it is a management/node/researcher peer
        - peer_id (str) : unique (within its `peer_type`) name for the peer
        - removeconf (bool, optional) : whether we should also remove the config file.
            Defaults to False.

    Raises:

    Returns:
        - None
    """

    # check arguments
    check_peer_args(peer_type, peer_id)
    if not isinstance(removeconf, bool):
        print(f"CRITICAL: bad type for `removeconf` {removeconf}")
        exit(1)

    # read peer config file
    filepath = os.path.join(peer_config_folder, peer_type, peer_id, config_file)
    if not os.path.isfile(filepath):
        # want explicit message and failure
        print(f"ERROR: do nothing, peer {peer_type}/{peer_id} does not exist. "
              "You need to create it with `genconf` first.")
        return

    peer_config = read_config_file(filepath)
    if 'VPN_IP' not in peer_config:
        print("CRITICAL: missing entry in peer config file : `VPN_IP`")
        exit(1) 

    # remove peer declared with `peer_id`'s IP prefix from current wireguard configuration
    current_peers = get_current_peers()
    try:
        remove_peer = ip.IPv4Network(f"{peer_config['VPN_IP']}/32")
    except (ip.AddressValueError, ip.NetmaskValueError) as e:
        print(f"CRITICAL: bad `VPN_IP` {peer_config['VPN_IP']} gives error : {e}")
        exit(1)

    for peer in current_peers:
        if peer[1] == str(remove_peer):
            try:
                subprocess.run(
                    ["wg", "set", "wg0", "peer", peer[0], "remove"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"CRITICAL: removing peer from wireguard interface failed with error : {e}")
                exit(1)

            print(f"info: successfully removed peer {peer[0]}")

    # save updated wireguard config file
    save_wg_file()

    if removeconf is True:
        remove_config_file(peer_type, peer_id)


def list() -> None:
    """ Build cross information on peers from configuration files and
    current wireguard interface configuration, pretty print on standard output

    Args:

    Raises:

    Returns:
        - None
    """

    # Structure of `peers`
    # peers = {
    #   IP_prefix_1 = {
    #       'name' = str(name_of_peer_1)
    #       'publickeys = [ str(peer_1_key_A), ... ]
    #   ...
    #   }
    # }
    peers = {}

    # scan peer config files
    for peer_type in os.listdir(peer_config_folder):
        for peer_id in os.listdir(os.path.join(peer_config_folder, peer_type)):

            filepath = os.path.join(peer_config_folder, peer_type, peer_id, config_file)
            peer_config = read_config_file(filepath)

            peer_tmpconf = { 'type': peer_type, 'id': peer_id }
            peer_tmpconf['publickeys'] = []
            try:
                peers[str(ip.IPv4Network(f"{peer_config['VPN_IP']}/32"))] = peer_tmpconf
            except (ip.AddressValueError, ip.NetmaskValueError) as e:
                print(f"CRITICAL: bad `VPN_IP` {peer_config['VPN_IP']} gives error : {e}")
                exit(1)

    # scan active peers list
    current_peers = get_current_peers()
    for peer_declared in current_peers:
        for pkey, pval in peers.items():
            if pkey == peer_declared[1]:
                pval['publickeys'].append(peer_declared[0])
                break
        if not peer_declared[1] in peers:
            peer_tmpconf = { 'type': '?', 'id': '?' }
            peer_tmpconf['publickeys'] = [ peer_declared[0] ]
            peers[peer_declared[1]] = peer_tmpconf

    # display result
    pretty_peers = [[v['type'], v['id'], k, v['publickeys']] for k, v in peers.items()]
    print(tabulate.tabulate(pretty_peers, headers = ['type', 'id', 'prefix', 'peers']))


#
# Main 
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure Wireguard peers on the server")
    subparsers = parser.add_subparsers(dest='operation', required=True, help="operation to perform")

    parser_genconf = subparsers.add_parser("genconf", help="generate the config file for a new peer")
    parser_genconf.add_argument(
        "type",
        choices=peer_types,
        help="type of client to generate config for")
    parser_genconf.add_argument("id", type=str, help="id of the new peer")


    parser_add = subparsers.add_parser("add", help="add a new peer")
    parser_add.add_argument(
        "type",
        choices=peer_types,
        help="type of client to add")
    parser_add.add_argument("id", type=str, help="id of the client")
    parser_add.add_argument("publickey", type=str, help="publickey of the client")

    parser_remove = subparsers.add_parser("remove", help="remove a peer")
    parser_remove.add_argument(
        "type",
        choices=peer_types,
        help="type of client to remove")
    parser_remove.add_argument("id", type=str, help="id client to remove")

    parser_remove = subparsers.add_parser("removeconf", help="remove a peer and its config file")
    parser_remove.add_argument(
        "type",
        choices=peer_types,
        help="type of client to remove")
    parser_remove.add_argument("id", type=str, help="id client to remove")

    parser_list = subparsers.add_parser("list", help="list peers and config files")

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
        list()
