# WIP notes for Fed-BioMed VPN'ization

To be converted to updates for main README.md + additions to install scripts

## building images

```bash
cd ./envs/vpn/docker
# CONTAINER_UID and CONTAINER_GID are target id for running the container
# TODO: check if we can use different id than the account building the images
#
# final configuration will be : build all containers
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build
# WIP configuration is : build existing containers
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build vpnserver
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn) docker-compose build node
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build mqtt
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) CONTAINER_USER=$(id -un) CONTAINER_GROUP=$(id -gn)  docker-compose build restful
```

## launching containers

* on the vpn server / node / mqtt server / restful
```bash
cd ./envs/vpn/docker
docker-compose up -d vpnserver
docker-compose up -d node
docker-compose up -d mqtt
docker-compose up -d restful
```

## connecting to containers

Can connect to a container only if the corresponding container is already running

* connect on the VPN server / node / mqtt server / restful as root to configure the VPN
```bash
docker container exec -ti fedbiomed-vpn-vpnserver bash
docker container exec -ti fedbiomed-vpn-node bash
docker container exec -ti fedbiomed-vpn-mqtt bash
docker container exec -ti fedbiomed-vpn-restful bash
```
* connect on the node as user to handle experiments
```bash
docker container exec -ti -u $(id -u) fedbiomed-vpn-node bash
```


## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/
