# Running Docker Images Behind a WireGuard VPN

Authentication between Fed-BioMed components is provided by [mutual authentication (mTLS)](./mutual-tls.md), inside a VPN or not. This page shows how to encapsulate the federated network in a VPN, for deployments where extra protection is necessary. The Fed-BioMed Docker images do not include a VPN tool by default; instead, a VPN client can run in a container that shares its network with the Fed-BioMed container, or on the host itself. This example uses WireGuard, but any other VPN server and client can be used.

This setup places the researcher and every node in a private [WireGuard](https://www.wireguard.com/) network, so the researcher gRPC port is reachable only from inside the VPN. It reuses the standard `fedbiomed/researcher` and `fedbiomed/node` images, unchanged, and requires two additional third-party components:

- **VPN server**: [wg-easy](https://github.com/wg-easy/wg-easy), a WireGuard server with a web UI, running on a host that every site can reach.
- **VPN client**: a sidecar container per site, using [linuxserver/wireguard](https://docs.linuxserver.io/images/docker-wireguard/). The Fed-BioMed container joins the sidecar network with `network_mode: "service:wg-client"`, so all its traffic goes through the sidecar network interfaces, including the VPN one.

The examples assume the VPN subnet `10.8.0.0/24` and the researcher peer at `10.8.0.2`. Use the subnet and addresses shown in the wg-easy admin UI.

## VPN server

This setup must include a VPN server, deployed and maintained separately from the Fed-BioMed components.

On the VPN server host, create a `docker-compose.yml` that runs the WireGuard server for the whole federation:

```yaml
services:
  wg-easy:                             # WireGuard server and its admin UI
    image: ghcr.io/wg-easy/wg-easy:15.5.0
    container_name: wg-easy
    environment:
      - INIT_ENABLED=true
      - INIT_USERNAME=admin
      - INIT_PASSWORD=<long-admin-password>
      - INIT_HOST=<vpn-server-public-ip-or-dns-name>
      - INIT_PORT=51820
      - DISABLE_IPV6=true
      - INSECURE=true                  # admin UI over HTTP, only published on this host
    volumes:
      - wg-easy:/etc/wireguard
      - /lib/modules:/lib/modules:ro
    ports:
      - "51820:51820/udp"              # WireGuard, reachable by every site
      - "127.0.0.1:51821:51821/tcp"    # admin UI, this host only
    cap_add:
      - NET_ADMIN
      - SYS_MODULE
    sysctls:
      - net.ipv4.ip_forward=1
      - net.ipv4.conf.all.src_valid_mark=1
    restart: unless-stopped

volumes:
  wg-easy:                             # server keys and peer list
```

Start it with `docker compose up -d`, then remove the `INIT_*` variables: they are only used on first start.

Open the admin UI through an SSH tunnel from your machine:

```bash
ssh -L 51821:localhost:51821 <user>@<vpn-server>
```

then browse to `http://localhost:51821`. Create one peer for the researcher first (it receives `10.8.0.2`), then one peer for every node, and download each peer `.conf` file. In each file, set `AllowedIPs` to the VPN subnet only:

```ini
[Peer]
AllowedIPs = 10.8.0.0/24
```

## Researcher site

On the researcher host, save the researcher peer file as `wireguard/wg_confs/wg0.conf`, next to this `docker-compose.yml`. It runs the researcher behind a VPN client sidecar:

```yaml
services:
  wg-client:                           # VPN client, connects the site to the VPN server
    image: lscr.io/linuxserver/wireguard:1.0.20260223-r0-ls128
    cap_add:
      - NET_ADMIN
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=Etc/UTC
    volumes:
      - ./wireguard:/config
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    ports:
      - "8888:8888"                    # Jupyter, published on the sidecar
    restart: unless-stopped

  researcher:                          # Fed-BioMed researcher, on the sidecar network
    image: fedbiomed/researcher:latest
    network_mode: "service:wg-client"
    environment:
      - FBM_SERVER_HOST=0.0.0.0        # default, also listens on the VPN interface
      - FBM_SERVER_PORT=50051
    volumes:
      - fbm-researcher:/fbm-researcher

volumes:
  fbm-researcher:                      # researcher component directory
```

The researcher shares the sidecar network, so its ports can only be published on the sidecar. Port `50051` is not published: nodes reach it through the VPN.

## Node site

On each node host, save that node's peer file as `wireguard/wg_confs/wg0.conf`, next to this `docker-compose.yml`. It runs the node behind a VPN client sidecar:

```yaml
services:
  wg-client:                           # VPN client, connects the site to the VPN server
    image: lscr.io/linuxserver/wireguard:1.0.20260223-r0-ls128
    cap_add:
      - NET_ADMIN
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=Etc/UTC
    volumes:
      - ./wireguard:/config
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    ports:
      - "8484:8484"                    # Node GUI, published on the sidecar
    restart: unless-stopped

  node:                                # Fed-BioMed node, on the sidecar network
    image: fedbiomed/node:latest
    network_mode: "service:wg-client"
    environment:
      - FBM_RESEARCHER_IP=10.8.0.2     # researcher VPN IP
      - FBM_RESEARCHER_PORT=50051
    volumes:
      - fbm-node:/fbm-node

volumes:
  fbm-node:                            # node component directory
```

Start each site with `docker compose up -d`. If the `wg-client` container is recreated, recreate the Fed-BioMed container too, as it loses its network otherwise.

## Quick check

On the VPN server, every peer should show a recent handshake:

```bash
docker exec wg-easy wg show
```

On a node site, check the tunnel and that the researcher answers through it:

```bash
docker compose exec wg-client wg show
docker compose exec wg-client ping -c 3 10.8.0.2
```

In the researcher Jupyter (`http://localhost:8888` on the researcher host), list the datasets of the connected nodes. Each node started with the default MNIST dataset shows up:

```python
from fedbiomed.researcher.config import config
from fedbiomed.researcher.requests import Requests

Requests(config).list(verbose=True)
```

If a node is missing, check its connection state with `fedbiomed node status`, as described in [Checking the Researcher Connection](./docker.md#checking-the-researcher-connection).

## Tips

- **Set `AllowedIPs` to the VPN subnet only.** Only Fed-BioMed traffic then goes through the VPN; the rest of the site traffic, such as package downloads, keeps its usual route.
- **Keep the wg-easy admin UI private.** Publish it on `127.0.0.1` only, reach it through an SSH tunnel, and never expose port `51821` publicly.
- **Pin the version of each third-party image.** Use an exact tag, as in the examples above, and update it deliberately after checking the release notes.
