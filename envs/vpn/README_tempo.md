# WIP notes for Fed-BioMed VPN'ization

To be converted to updates for main README.md + additions to install scripts

## building images

```bash
cd ./envs/vpn/docker
# CONTAINER_UID and CONTAINER_GID are target id for running the container
# TODO: check if we can use different id than the account building the images
#
# final configuration will be : build all containers
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) docker-compose build
# WIP configuration is : build existing containers
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) docker-compose build vpnserver
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) docker-compose build node
```

## launching containers

* on the vpn server
```bash
cd ./envs/vpn/docker
# launch wanted container(s)
docker-compose up -d vpnserver
docker-compose up -d node
```


## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/
