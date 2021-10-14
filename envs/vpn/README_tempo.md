# WIP notes for Fed-BioMed VPN'ization

**TODO** : convert to updates for main README.md + additions to install scripts

Which identity to use ?

- outside of containers : work with the user account used for launching the containers (eg `[user@node $]`), unless this is explicitely written to work as root (eg `[root@node #]`). This is also the account used by the containers when they drop privileges.
- inside containers : follow instructions, work as root for managing vpn (eg `[root@vpnserver-container #]`), work as user for managing experiments (eg `[user@node-container $]`)

Which machine to use ?

- when using distinct machines for the components, type node commands (eg `[user@node $]`) on node, researcher commands (eg `[user@researcher $]`) on researcher, network commands on network (eg `[user@network $]`)
- when running all components on a single laptop, all components are on this machine

## containers

### building images

Done when initializing each container (see after)

```bash
#[user@laptop $] cd ./envs/vpn/docker
## CONTAINER_{UID,GID,USER,GROUP} are target id for running the container
## **TODO**: check if we can use different id than the account building the images
#
## when running on a single machine : build all containers at one time with
#[user@laptop $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build
```

Caveat : docker >= 20.10.0 needed to build mqtt, see [there](https://wiki.alpinelinux.org/wiki/Release_Notes_for_Alpine_3.14.0#faccessat2). With older docker version it fails with a `make: sh: Operation not permitted`


## setup VPN and fedbiomed

### initializing vpnserver

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build base
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build vpnserver
```
* set the VPN server public IP *VPN_SERVER_PUBLIC_ADDR*
```bash
[user@network $] vi ./vpnserver/run_mounts/config/config.env # change VPN_SERVER_PUBLIC_ADDR
```
* launch container
```bash
[user@network $] docker-compose up -d vpnserver
```
* connect and generate config for components
```bash
[user@network $] docker-compose exec vpnserver bash
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf management mqtt
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf management restful
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf node node1
[root@vpnserver-container #] python ./vpn/bin/configure_peer.py genconf researcher researcher1
```

Run this for all launches of the container :
* launch container
```bash
[user@network $] docker-compose up -d vpnserver
```

### initializing mqtt

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build mqtt
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] cp ./vpnserver/run_mounts/config/config_peers/management/mqtt/config.env ./mqtt/run_mounts/config/config.env
```
* launch container
```bash
[user@network $] docker-compose up -d mqtt
```
* retrieve the *publickey*
```bash
[user@network $] docker-compose exec mqtt wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add management mqtt *publickey*
## other option :
#[user@network $] docker-compose exec vpnserver bash
#[root@vpnserver-container #] python ./vpn/bin/configure_peer.py add management mqtt *publickey*
```

Run this for all launches of the container :
* launch container
```bash
[user@network $] docker-compose up -d mqtt
```

### initializing restful

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build restful
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@network $] cd ./envs/vpn/docker
[user@network $] cp ./vpnserver/run_mounts/config/config_peers/management/restful/config.env ./restful/run_mounts/config/config.env
```
* launch container
```bash
[user@network $] docker-compose up -d restful
```
* retrieve the *publickey*
```bash
[user@network $] docker-compose exec restful wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add management restful *publickey*
```

Run this for all launches of the container :
* launch container
```bash
[user@network $] docker-compose up -d restful
```

### initializing node

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build base
[user@node $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build node
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@node $] cd ./envs/vpn/docker
[user@node $] vi ./node/run_mounts/config/config.env
# add the content of the command below in the edited file above
[user@network $] cat ./vpnserver/run_mounts/config/config_peers/node/node1/config.env
## if running on a single machine
#[user@laptop $] cp ./vpnserver/run_mounts/config/config_peers/node/node1/config.env ./node/run_mounts/config/config.env
```
* launch container
```bash
[user@node $] docker-compose up -d node
```
* retrieve the *publickey*
```bash
[user@node $] docker-compose exec node wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@network $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add node node1 *publickey*
```

Run this for all launches of the container :

* launch container
```bash
[user@node $] docker-compose up -d node
```
* TODO: better package/scripting needed
  Connect again to the node and launch manually, now that the VPN is established
```bash
[user@laptop $] docker-compose exec -u $(id -u) node bash
# TODO : make more general by including it in the VPN configuration and user environment ?
# TODO : create scripts in VPN environment
# need proper parameters at first launch to create configuration file
[user@node-container $] export MQTT_BROKER=10.220.0.2
[user@node-container $] export MQTT_BROKER_PORT=1883
[user@node-container $] export UPLOADS_URL="http://10.220.0.3:8000/upload/"
[user@node-container $] export PYTHONPATH=/fedbiomed
[user@node-container $] eval "$(conda shell.bash hook)"
[user@node-container $] conda activate fedbiomed-node
# example : add MNIST dataset using persistent (mounted) /data
[user@node-container $] python -m fedbiomed.node.cli -am /data
# start the node
[user@node-container $] python -m fedbiomed.node.cli --start
```

### initializing researcher

Run this only at first launch of container or after cleaning :

* build container
```bash
[user@researcher $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build base
[user@researcher $] CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build researcher
```
* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
[user@laptop $] cd ./envs/vpn/docker
[user@node $] vi ./researcher/run_mounts/config/config.env
# add the content of the command below in the edited file above
[user@network $] cat ./vpnserver/run_mounts/config/config_peers/researcher/researcher1/config.env
## if running on a single machine
#[user@laptop $] cp ./vpnserver/run_mounts/config/config_peers/node/node1/config.env ./researcher/run_mounts/config/config.env
```
* launch container
```bash
[user@laptop $] docker-compose up -d researcher
```
* retrieve the *publickey*
```bash
[user@laptop $] docker-compose exec researcher wg show wg0 public-key
```
* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
[user@laptop $] docker-compose exec vpnserver python ./vpn/bin/configure_peer.py add researcher researcher1 *publickey*
```

Run this for all launches of the container :

* TODO: better package/scripting needed
  Connect again to the researcher and launch manually, now that the VPN is established
```bash
[user@laptop $] docker-compose exec -u $(id -u) researcher bash
# TODO : make more general by including it in the VPN configuration and user environment ?
# TODO : create scripts in VPN environment
# need proper parameters at first launch to create configuration file
[user@researcher-container $] export MQTT_BROKER=10.220.0.2
[user@researcher-container $] export MQTT_BROKER_PORT=1883
[user@researcher-container $] export UPLOADS_URL="http://10.220.0.3:8000/upload/"
[user@researcher-container $] export PYTHONPATH=/fedbiomed
[user@researcher-container $] eval "$(conda shell.bash hook)"
[user@researcher-container $] conda activate fedbiomed-researcher
# ... or any other command
[user@researcher-container $] ./notebooks/getting-started.py
```

* to use notebooks, from outside the researcher container connect to `http://localhost:8888` or `http://SERVER_IP:8888`
  * TODO : add protection for distant connection to researcher


## connecting to containers

You can connect to a container only if the corresponding container is already running

* connect on the VPN server / node / mqtt server / restful as root to configure the VPN
```bash
[user@network $] docker-compose exec vpnserver bash
[user@network $] docker-compose exec mqtt bash
[user@network $] docker-compose exec restful bash
[user@node $] docker-compose exec node bash
[user@researcher $] docker-compose exec researcher bash
```
* connect on the node as user to handle experiments
```bash
[user@node $] docker-compose exec -ti -u $(id -u) node bash
[user@researcher $] docker-compose exec -ti -u $(id -u) researcher bash
```

Note : can also use commands in the form
```bash
[user@node $] docker container exec -ti -u $(id -u) fedbiomed-vpn-node bash
```

## cleaning

### vpnserver

```bash
cd ./envs/vpn/docker
docker-compose rm -sf vpnserver
# currently as root 
# TODO write config files as CONTAINER_USER
rm -rf vpnserver/run_mounts/config/{config_peers,ip_assign,wireguard}
```

### mqtt

```
cd ./envs/vpn/docker
docker-compose rm -sf mqtt
rm -rf ./mqtt/run_mounts/config/wireguard
echo > ./mqtt/run_mounts/config/config.env
```

### restful

```
cd ./envs/vpn/docker
docker-compose rm -sf restful
rm -rf ./restful/run_mounts/config/wireguard
echo > ./restful/run_mounts/config/config.env

rm -f ./restful/run_mounts/app/db.sqlite3
# also clean saved files ? (same for env/developement)
```

### node 

```
cd ./envs/vpn/docker
docker-compose rm -sf node
rm -rf ./node/run_mounts/config/wireguard
echo > ./node/run_mounts/config/config.env
rm -rf ./node/run_mounts/data/*
```

### researcher

Same as node

```
cd ./envs/vpn/docker
docker-compose rm -sf researcher
rm -rf ./researcher/run_mounts/config/wireguard
echo > ./researcher/run_mounts/config/config.env
rm -rf ./researcher/run_mounts/data/*
```

## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/
