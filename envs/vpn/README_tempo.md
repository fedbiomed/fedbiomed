# WIP notes for Fed-BioMed VPN'ization

To be converted to updates for main README.md + additions to install scripts

## building base image

```bash
docker image build -t fedbiomed/base ./envs/vpn/docker/base
```


## background / wireguard

* boringtun - wireguard userspace rust implementation
    - https://blog.cloudflare.com/boringtun-userspace-wireguard-rust/
    - https://github.com/cloudflare/boringtun

* wireguard tools : https://git.zx2c4.com/wireguard-tools/about/

* wireguard vs openvpn : https://restoreprivacy.com/vpn/wireguard-vs-openvpn/
