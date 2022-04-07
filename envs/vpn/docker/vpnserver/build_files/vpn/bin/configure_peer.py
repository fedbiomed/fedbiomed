import argparse
from string import Template
import subprocess
import os
import re
import ipaddress as ip
import tabulate

#
# Script for handling wireguard VPN peer configurations
# - launched in `vpnserver` container (Linux), thus can use some os-specific commands
#

TEMPLATE_FILE="/fedbiomed/vpn/config_templates/config_%s.env"
ASSIGN_CONFIG_FILE="/config/ip_assign/last_ip_assign_%s"
PEER_CONFIG_FOLDER="/config/config_peers"


def genconf(peer_type, peer_id):
    assert peer_type=="researcher" or peer_type=="node" or peer_type=="management"

    # wireguard keys
    peer_psk=subprocess.run(["wg", "genpsk"], stdout=subprocess.PIPE, text=True).stdout.rstrip('\n')
    server_public_key=subprocess.run(["wg", "show", "wg0", "public-key"], stdout=subprocess.PIPE, text=True).stdout.rstrip('\n')
   
    # assign the next available ip to the peer
    vpn_net=ip.IPv4Interface(f"{os.environ['VPN_IP']}/{os.environ['VPN_SUBNET_PREFIX']}").network

    assign_net=ip.IPv4Network(os.environ[f"VPN_{peer_type.upper()}_IP_ASSIGN"])

    assign_file=ASSIGN_CONFIG_FILE%peer_type
    if os.path.exists(assign_file) and os.path.getsize(assign_file) > 0:
        os.setegid(container_gid)
        os.seteuid(container_uid)
        f= open(assign_file, 'r+')
        os.seteuid(init_uid)
        os.setegid(init_gid)

        peer_addr_ip=ip.IPv4Address(f.read())+1
        f.seek(0)
        f.write(str(peer_addr_ip))
        f.close()
    else:
        os.setegid(container_gid)
        os.seteuid(container_uid)
        f = open(assign_file, 'w')
        os.seteuid(init_uid)
        os.setegid(init_gid)

        peer_addr_ip=assign_net.network_address+2
        f.write(str(peer_addr_ip))
        f.close()
    

    assert peer_addr_ip in assign_net
    assert peer_addr_ip in vpn_net

    # create peer configuration
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

    assert None not in mapping.values()
    assert "" not in mapping.values()

    os.setegid(container_gid)
    os.seteuid(container_uid)
    outpath=f"{PEER_CONFIG_FOLDER}/{peer_type}"
    os.makedirs(outpath, exist_ok=True)
    outpath+=f"/{peer_id}"
    os.mkdir(outpath)
    filepath=f"{outpath}/config.env"
    
    f_config = open(filepath, 'w')
    os.seteuid(init_uid)
    os.setegid(init_gid)

    with open(TEMPLATE_FILE%peer_type, 'r') as f_template:
        f_config.write(Template(f_template.read()).substitute(mapping))
    f_config.close()

    print("Configuration generated in", filepath)


def add(peer_type, peer_id, peer_public_key):
    assert peer_type=="researcher" or peer_type=="node" or peer_type=="management"
    
    filepath=f"{PEER_CONFIG_FOLDER}/{peer_type}/{peer_id}/config.env"

    with open(filepath, 'r') as f:
        peer_config=dict(tuple(line.removeprefix('export').lstrip().split('=', 1)) for line in map(lambda line: line.strip(" \n"), f.readlines()) if not line.startswith('#') and not line=='')

    # apply the config to the server
    subprocess.run(["wg", "set", "wg0", "peer", peer_public_key, "allowed-ips", f"{peer_config['VPN_IP']}/32", "preshared-key", "/dev/stdin"], text=True, input=peer_config['VPN_SERVER_PSK']) 
    subprocess.run(["bash", "-c", "(umask 0077; wg showconf wg0 > /config/wireguard/wg0.conf)"])

def remove(peer_type, peer_id, removeconf: bool = False):
    assert peer_type=="researcher" or peer_type=="node" or peer_type=="management"

    filepath=f"{PEER_CONFIG_FOLDER}/{peer_type}/{peer_id}/config.env"

    with open(filepath, 'r') as f:
        peer_config=dict(tuple(line.removeprefix('export').lstrip().split('=', 1)) for line in map(lambda line: line.strip(" \n"), f.readlines()) if not line.startswith('#') and not line=='')
   
    # same as add() up to there - to be factored

    f = os.popen('wg show wg0 allowed-ips')
    for line in f:
        peer = re.split('\s+', line.strip(" \n"))
        if peer[1] == f"{peer_config['VPN_IP']}/32":
            subprocess.run(["wg", "set", "wg0", "peer", peer[0], "remove"])
            print(f"info: removed peer {peer[0]}")
    f.close()

    # same as add - to be factored
    subprocess.run(["bash", "-c", "(umask 0077; wg showconf wg0 > /config/wireguard/wg0.conf)"])

    if removeconf is True:
        conf_dir = f"{PEER_CONFIG_FOLDER}/{peer_type}/{peer_id}"
        conf_file = f"{conf_dir}/config.env"
        if os.path.isdir(conf_dir) and os.path.isfile(conf_file):
            os.setegid(container_gid)
            os.seteuid(container_uid)
            os.remove(conf_file)
            os.rmdir(conf_dir)
            print(f"info: removed config dir {conf_dir}")
        else:
            print("CRITICAL: missing configuration file {conf_file}")
            exit(1)

def list():

    # peers = {
    #   IP_prefix_1 = {
    #       'name' = str(name_of_peer_1)
    #       'publickeys = [ str(peer_1_key_A), ... ]
    #   ...
    #   }
    # }
    peers = {}

    # scan peer config files
    for peer_type in os.listdir(PEER_CONFIG_FOLDER):
        for peer_id in os.listdir(os.path.join(PEER_CONFIG_FOLDER, peer_type)):
            peer_tmpconf = {}

            filepath = os.path.join(PEER_CONFIG_FOLDER, peer_type, peer_id, 'config.env')
            # to be factored with add/remove
            with open(filepath, 'r') as f:
                peer_config=dict(tuple(line.removeprefix('export').lstrip().split('=', 1)) for line in map(lambda line: line.strip(" \n"), f.readlines()) if not line.startswith('#') and not line=='')

            peer_tmpconf['name'] = f'{peer_type}/{peer_id}'
            peer_tmpconf['publickeys'] = []
            peers[f"{peer_config['VPN_IP']}/32"] = peer_tmpconf

    # scan active peers list
    
    # (partial) same as `remove` - to be factored
    f = os.popen('wg show wg0 allowed-ips')
    for line in f:
        peer_declared = re.split('\s+', line.strip(" \n"))

        for pkey, pval in peers.items():
            if pkey == peer_declared[1]:
                pval['publickeys'].append(peer_declared[0])
                break
        if not peer_declared[1] in peers:
            peer_tmpconf = {}
            peer_tmpconf['name'] = '?/?'
            peer_tmpconf['publickeys'] = [ peer_declared[0] ]
            peers[peer_declared[1]] = peer_tmpconf
    f.close()

    # display result
    pretty_peers = [[v['name'], k, v['publickeys']] for k,v in peers.items()]
    print(tabulate.tabulate(pretty_peers, headers = ['name', 'prefix', 'peers']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure Wireguard peers on the server")
    subparsers = parser.add_subparsers(dest='operation', required=True, help="operation to perform")

    parser_genconf = subparsers.add_parser("genconf", help="generate the config file for a new peer")
    parser_genconf.add_argument("type", choices=["researcher", "node", "management"], help="type of client to generate config for")
    parser_genconf.add_argument("id", type=str, help="id of the new peer")


    parser_add = subparsers.add_parser("add", help="add a new peer")
    parser_add.add_argument("type", choices=["researcher", "node", "management"], help="type of client to add")
    parser_add.add_argument("id", type=str, help="id of the client")
    parser_add.add_argument("publickey", type=str, help="publickey of the client")

    parser_remove = subparsers.add_parser("remove", help="remove a peer")
    parser_remove.add_argument("type", choices=["researcher", "node", "management"], help="type of client to remove")
    parser_remove.add_argument("id", type=str, help="id client to remove")

    parser_remove = subparsers.add_parser("removeconf", help="remove a peer and its config file")
    parser_remove.add_argument("type", choices=["researcher", "node", "management"], help="type of client to remove")
    parser_remove.add_argument("id", type=str, help="id client to remove")

    parser_list = subparsers.add_parser("list", help="list peers and config files")

    args = parser.parse_args()


    init_uid = os.geteuid()
    if 'CONTAINER_UID' in os.environ:
        container_uid = int(os.environ['CONTAINER_UID'])
    else:
        container_uid = init_uid

    init_gid = os.getegid()
    if 'CONTAINER_GID' in os.environ:
        container_gid = int(os.environ['CONTAINER_GID'])
    else:
        container_gid = init_gid

    if args.operation=="genconf":
        genconf(args.type, args.id)
    elif args.operation=="add":
        add(args.type, args.id, args.publickey)
    elif args.operation=="remove":
        remove(args.type, args.id)
    elif args.operation=="removeconf":
        remove(args.type, args.id, True)
    elif args.operation=="list":
        list()
    
