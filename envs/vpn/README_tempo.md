# WIP notes for Fed-BioMed VPN'ization

To be converted to updates for main README.md + additions to install scripts

## building images

```bash
cd ./envs/vpn/docker
CONTAINER_UID=$(id -u) CONTAINER_GID=$(id -g) docker-compose build
```

## launching containers

* on the vpn server
```bash
cd ./envs/vpn/docker
docker-compose up -d vpnserver
```


## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/
