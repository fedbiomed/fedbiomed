# Running Docker Images Behind a WireGuard VPN

Encryption and authentication between Fed-BioMed components are provided by [mutual authentication (mTLS)](./mutual-tls.md), inside a VPN or not. This page shows how to encapsulate the federated network in a VPN, as an optional additional layer of protection. The Fed-BioMed Docker images do not include a VPN tool by default; instead, a VPN client can run in a container that shares its network with the Fed-BioMed container, or on the host itself. This example uses WireGuard, but any other VPN server and client can be used.

This setup places the researcher and every node in a private [WireGuard](https://www.wireguard.com/) network, so the researcher gRPC port is reachable from other sites only through the VPN. It reuses the standard `fedbiomed/researcher` and `fedbiomed/node` images, unchanged, and requires two additional third-party components:

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
      - INIT_PASSWORD=<long-admin-password>   # admin UI password
      - INIT_HOST=<vpn-server-public-ip-or-dns-name>   # address every site uses to reach this host
      - INIT_PORT=51820
      - DISABLE_IPV6=true
      - INSECURE=true                  # admin UI over HTTP, only published on this host
    volumes:
      - wg-easy:/etc/wireguard
      - /lib/modules:/lib/modules:ro
    ports:
      - "51820:51820/udp"              # WireGuard, reachable by every site
      - "127.0.0.1:51821:51821/tcp"    # admin UI, this host only
    cap_add:                           # create the WireGuard interface
      - NET_ADMIN
      - SYS_MODULE
    sysctls:                           # route traffic between peers
      - net.ipv4.ip_forward=1
      - net.ipv4.conf.all.src_valid_mark=1
    restart: unless-stopped

volumes:
  wg-easy:                             # server keys and peer list
```

Start it with `docker compose up -d`. The `INIT_*` variables are only read on first start; later changes to these settings are made in the admin UI.

Open the admin UI through an SSH tunnel from your machine:

```bash
ssh -L 51821:localhost:51821 <user>@<vpn-server>
```

then browse to `http://localhost:51821`. Create one peer for the researcher first (it receives `10.8.0.2`), then one peer for every node, and download each peer `.conf` file. In each file, set `AllowedIPs` to the VPN subnet only and remove the `DNS` line, so that only Fed-BioMed traffic goes through the VPN; the rest of the site traffic, such as package downloads and name resolution, keeps its usual route. Also set `PersistentKeepalive`, so that each peer, including the researcher which only answers requests, stays reachable through the VPN server:

```ini
[Peer]
AllowedIPs = 10.8.0.0/24
PersistentKeepalive = 25
```

## Researcher site

On the researcher host, save the researcher peer file as `wireguard/wg_confs/wg0.conf`, next to this `docker-compose.yml`. It runs the researcher behind a VPN client sidecar:

- `wg-client` connects the site to the VPN with the researcher peer file.
- `researcher` uses the network of `wg-client`, so nodes reach it at its VPN address `10.8.0.2`.

```yaml
services:
  wg-client:                           # VPN client, connects the site to the VPN server
    image: lscr.io/linuxserver/wireguard:1.0.20260223-r0-ls128
    cap_add:
      - NET_ADMIN
    environment:
      - PUID=1000                      # user and group owning ./wireguard on the host
      - PGID=1000
      - TZ=Etc/UTC                     # container timezone
    volumes:
      - ./wireguard:/config            # holds wg_confs/wg0.conf, the peer file
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    ports:
      - "127.0.0.1:8888:8888"          # Jupyter (no token), this host only
    restart: unless-stopped

  researcher:                          # Fed-BioMed researcher, on the sidecar network
    # FBM_TAG: a published version (e.g. 6.4.1) is pulled from Docker Hub;
    # "latest" (default) is not published, so it must be built locally first
    image: fedbiomed/researcher:${FBM_TAG:-latest}
    network_mode: "service:wg-client"
    environment:
      - FBM_SERVER_HOST=0.0.0.0        # listens on all interfaces, including the VPN one
      - FBM_SERVER_PORT=50051
    volumes:
      - fbm-researcher:/fbm-researcher

volumes:
  fbm-researcher:                      # researcher component directory
```

The researcher shares the sidecar network, so its ports can only be published on the sidecar. Port `50051` is not published. Jupyter runs without a token, so it is published on `127.0.0.1` only; open it from your machine through an SSH tunnel:

```bash
ssh -L 8888:localhost:8888 <user>@<researcher-host>
```

then browse to `http://localhost:8888`. The node GUI is reached the same way, on port `8484` of the node host.

## Node site

On each node host, save that node's peer file as `wireguard/wg_confs/wg0.conf`, next to this `docker-compose.yml`. It runs the node behind a VPN client sidecar:

- `wg-client` is the same as on the researcher site, with that node's peer file.
- `node` uses the network of `wg-client`, and connects to the researcher at its VPN address, set in `FBM_RESEARCHER_IP`.

```yaml
services:
  wg-client:                           # VPN client, connects the site to the VPN server
    image: lscr.io/linuxserver/wireguard:1.0.20260223-r0-ls128
    cap_add:
      - NET_ADMIN
    environment:
      - PUID=1000                      # user and group owning ./wireguard on the host
      - PGID=1000
      - TZ=Etc/UTC                     # container timezone
    volumes:
      - ./wireguard:/config            # holds wg_confs/wg0.conf, the peer file
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    ports:
      - "127.0.0.1:8484:8484"          # Node GUI, this host only
    restart: unless-stopped

  node:                                # Fed-BioMed node, on the sidecar network
    # FBM_TAG: a published version (e.g. 6.4.1) is pulled from Docker Hub;
    # "latest" (default) is not published, so it must be built locally first
    image: fedbiomed/node:${FBM_TAG:-latest}
    network_mode: "service:wg-client"
    environment:
      - FBM_RESEARCHER_IP=10.8.0.2     # researcher VPN IP
      - FBM_RESEARCHER_PORT=50051
    volumes:
      - fbm-node:/fbm-node

volumes:
  fbm-node:                            # node component directory
```

Start each site with the Fed-BioMed version to run, the same on the researcher and every node:

```bash
FBM_TAG=<version-tag> docker compose up -d
```

`<version-tag>` is a version published on Docker Hub, such as `6.4.1`. Without `FBM_TAG`, the `latest` tag is used: it is not published on Docker Hub, so the image must first be [built from source](../../developer/docker-images.md) on that host. To avoid repeating the tag, write `FBM_TAG=<version-tag>` in a `.env` file next to the `docker-compose.yml`; `docker compose` reads it automatically.

If the `wg-client` container is recreated, recreate the Fed-BioMed container too, as it loses its network otherwise.

## Mutual authentication

The VPN does not replace [mutual authentication (mTLS)](./mutual-tls.md), which must be enabled on the researcher and every node:

- Each component has its certificate: it is generated when the component is created.
- The certificates are exchanged over a trusted channel: every node sends its certificate to the researcher, and the researcher sends its certificate to every node.
- The received certificates are registered: on the researcher, every node certificate; on each node, the researcher certificate.
- `mutual_authentication = True` is set in the `[authentication]` section of every component configuration, and each component is restarted.

See [Enabling mutual authentication](./mutual-tls.md#enabling-mutual-authentication) for the details of each step.

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

In the researcher Jupyter, opened through the SSH tunnel, list the datasets of the connected nodes. Each node started with the default MNIST dataset shows up:

```python
from fedbiomed.researcher.config import config
from fedbiomed.researcher.requests import Requests

Requests(config).list(verbose=True)
```

If a node is missing, check its connection state with `fedbiomed node status`, as described in [Checking the Researcher Connection](./docker.md#checking-the-researcher-connection).

## Tips

- **Keep the web interfaces private.** Publish the wg-easy admin UI, Jupyter and the node GUI on `127.0.0.1` only, reach them through an SSH tunnel, and never expose ports `51821`, `8888` or `8484` publicly.
- **Pin the version of each third-party image.** Use an exact tag, as in the examples above, and update it deliberately after checking the release notes.
