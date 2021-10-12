# WIP notes for Fed-BioMed VPN'ization

TODO : convert to updates for main README.md + additions to install scripts

- outside of containers : work with your user account, unless this is explicitely written to work as root
- inside containers : follow instructions (work as root for managing vpn, work as user for managing experiments)

## containers

### building images

```bash
cd ./envs/vpn/docker
# CONTAINER_{UID,GID,USER,GROUP} are target id for running the container
# TODO: check if we can use different id than the account building the images
#
# final configuration will be : build all containers
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build
# WIP configuration is : build existing containers
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build vpnserver
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build node
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build researcher
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build mqtt
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build restful
```

### launching containers

* on the vpn server / node / researcher / mqtt server / restful
```bash
cd ./envs/vpn/docker
docker-compose up -d vpnserver
docker-compose up -d node
docker-compose up -d researcher
docker-compose up -d mqtt
docker-compose up -d restful
```

### connecting to containers

Can connect to a container only if the corresponding container is already running

* connect on the VPN server / node / mqtt server / restful as root to configure the VPN
```bash
docker-compose exec vpnserver bash
docker-compose exec node bash
docker-compose exec researcher bash
docker-compose exec mqtt bash
docker-compose exec restful bash
```
* connect on the node as user to handle experiments
```bash
docker-compose exec -ti -u $(id -u) node bash
docker-compose exec -ti -u $(id -u) researcher bash
```

Note : can also use commands in the form
```bash
docker container exec -ti -u $(id -u) fedbiomed-vpn-node bash
```

## setup VPN and fedbiomed

### initializing vpnserver

* build and launch container
* set the VPN server public IP *VPN_SERVER_PUBLIC_ADDR* (can also do it inside container)
```bash
cd ./envs/vpn/docker
vi ./vpnserver/run_mounts/config/config.env # change VPN_SERVER_PUBLIC_ADDR
```
* connect and generate config for components
```bash
docker-compose exec vpnserver bash
python ./vpn/bin/configure_peer.py genconf management mqtt
python ./vpn/bin/configure_peer.py genconf management restful
python ./vpn/bin/configure_peer.py genconf node node1
python ./vpn/bin/configure_peer.py genconf researcher researcher1
```

### initializing mqtt

* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
cd ./envs/vpn/docker
cp ./vpnserver/run_mounts/config/config_peers/management/mqtt/config.env ./mqtt/run_mounts/config/config.env
```
* build and launch container
* retrieve the *publickey*
```bash
docker-compose exec mqtt wg show wg0 public-key
```

* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
docker-compose exec vpnserver bash
python ./vpn/bin/configure_peer.py add management mqtt *publickey*
```

### initializing restful

Basically same as mqtt with proper adaptations :

* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
cd ./envs/vpn/docker
cp ./vpnserver/run_mounts/config/config_peers/management/restful/config.env ./restful/run_mounts/config/config.env
```
* build and launch container
* retrieve the *publickey*
```bash
docker-compose exec restful wg show wg0 public-key
```

* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
docker-compose exec vpnserver bash
python ./vpn/bin/configure_peer.py add management restful *publickey*
```

### initializing node

* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
cd ./envs/vpn/docker
cp ./vpnserver/run_mounts/config/config_peers/node/node1/config.env ./node/run_mounts/config/config.env
```
* build and launch container
* retrieve the *publickey*
```bash
docker-compose exec node wg show wg0 public-key
```

* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
docker-compose exec vpnserver bash
python ./vpn/bin/configure_peer.py add node node1 *publickey*
```

* TODO: better package/scripting needed
  Connect again to the node and launch manually, now that the VPN is established
```bash
docker-compose exec -u $(id -u) node bash
# TODO : make more general by including it in the VPN configuration and user environment ?
# TODO : create scripts in VPN environment
# need proper parameters at first launch to create configuration file
export MQTT_BROKER=10.220.0.2
export MQTT_BROKER_PORT=1883
export UPLOADS_URL="http://10.220.0.3:8000/upload/"
export PYTHONPATH=/fedbiomed
eval "$(conda shell.bash hook)"
conda activate fedbiomed-node
python -m fedbiomed.node.cli --start
```

### initializing researcher

Same as node

* generate VPN client for this container (see above in vpnserver)
* configure the VPN client for this container
```bash
cd ./envs/vpn/docker
cp vpnserver/run_mounts/config/config_peers/researcher/researcher1/config.env researcher/run_mounts/config/config.env
```
* build and launch container
* retrieve the *publickey*
```bash
docker-compose exec researcher wg show wg0 public-key
```

* connect to the VPN server to declare the container as a VPN client with cut-paste of *publickey*
```bash
docker-compose exec vpnserver bash
python ./vpn/bin/configure_peer.py add researcher researcher1 *publickey*
```

* TODO: better package/scripting needed
  Connect again to the researcher and launch manually, now that the VPN is established
```bash
docker-compose exec -u $(id -u) researcher bash
# TODO : make more general by including it in the VPN configuration and user environment ?
# TODO : create scripts in VPN environment
# need proper parameters at first launch to create configuration file
export MQTT_BROKER=10.220.0.2
export MQTT_BROKER_PORT=1883
export UPLOADS_URL="http://10.220.0.3:8000/upload/"
export PYTHONPATH=/fedbiomed
eval "$(conda shell.bash hook)"
conda activate fedbiomed-researcher
# ... or any other command
./notebooks/getting-started.py
```

TODO : test with notebooks

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

Same as mqtt

```
cd ./envs/vpn/docker
docker-compose rm -sf node
rm -rf ./node/run_mounts/config/wireguard
echo > ./node/run_mounts/config/config.env
rm -rf ./node/run_mounts/data/*
```


## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/
