# Windows VPN Docker workflows

These PowerShell scripts replace the Linux-oriented `scripts/fedbiomed_vpn`
commands when running the VPN Docker setup from Windows Terminal.

Run all commands from PowerShell.

## One-time setup

```powershell
$env:COMPOSE_DIR = "C:\Users\adincer\Desktop\fedbiomed\envs\vpn\docker"
cd $env:COMPOSE_DIR
```

If PowerShell blocks script execution, run with `ExecutionPolicy Bypass`:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows_researcher_vpn_workflow.ps1
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows_node_vpn_workflow.ps1 -Target both
```

## Endpoint selection

By default, both scripts auto-detect a Windows host IPv4 address and use it as
the WireGuard endpoint:

```text
<auto-detected-host-ip>:51820
```

This is the Windows equivalent of a real deployment-style endpoint. The peer
containers reach the vpnserver through Docker's published UDP port:

```text
node/researcher -> Windows host IP:51820 -> vpnserver container:51820/udp
```

If autodetection picks the wrong address, list candidate IPv4 addresses:

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
  Select-Object InterfaceAlias,IPAddress
```

Then pass the desired address explicitly:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -VpnServerPublicAddr "<your-host-ip>"
```

You can also use IPv6 if reachable. Pass the raw IPv6 address; the script adds
WireGuard brackets automatically:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -VpnServerPublicAddr "fdc4:f303:9324::254"
```

This generates:

```text
[fdc4:f303:9324::254]:51820
```

## Researcher workflow

Default run: configure, start, register, and test researcher. It does not clean
or build unless requested.

```powershell
.\windows_researcher_vpn_workflow.ps1
```

With an explicit endpoint:

```powershell
.\windows_researcher_vpn_workflow.ps1 -VpnServerPublicAddr "<your-host-ip>"
```

Build images first:

```powershell
.\windows_researcher_vpn_workflow.ps1 -Build
```

Clean first:

```powershell
.\windows_researcher_vpn_workflow.ps1 -Clean
```

Clean and build first:

```powershell
.\windows_researcher_vpn_workflow.ps1 -Clean -Build
```

Remove the researcher container and exit:

```powershell
.\windows_researcher_vpn_workflow.ps1 -Down
```

Remove then recreate/start researcher:

```powershell
.\windows_researcher_vpn_workflow.ps1 -Down -Up
```

## Node workflow

Default run: configure, start, register, and test the selected node(s). It does
not clean or build unless requested.

First node:

```powershell
.\windows_node_vpn_workflow.ps1 -Target node
```

Second node:

```powershell
.\windows_node_vpn_workflow.ps1 -Target node2
```

Both nodes:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both
```

With an explicit endpoint:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -VpnServerPublicAddr "<your-host-ip>"
```

Build the shared node image first:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -Build
```

Clean selected node config first:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -Clean
```

Clean and build first:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -Clean -Build
```

Remove selected node container(s) and exit:

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -Down
```

Remove then recreate/start selected node container(s):

```powershell
.\windows_node_vpn_workflow.ps1 -Target both -Down -Up
```

## Script parameters

### Common parameters

`-VpnServerPublicAddr`

Host/IP peers should use to reach the vpnserver before the VPN exists. If not
provided, the script auto-detects a Windows host IPv4 address. Examples:

```powershell
-VpnServerPublicAddr "192.168.1.50"
-VpnServerPublicAddr "my-vpn-host.example.org"
-VpnServerPublicAddr "fdc4:f303:9324::254"
```

`-VersionTag`

Docker image tag. Default:

```text
6.3.3
```

`-InstanceId`

Container/network suffix. Default:

```text
default
```

`-ContainerUid`, `-ContainerGid`, `-ContainerUser`, `-ContainerGroup`

Runtime identity injected into Docker Compose. Defaults are:

```text
1000 / 1000 / current Windows username / current Windows username
```

`-ComposeDir`

Path to `envs/vpn/docker`. Defaults to `$env:COMPOSE_DIR`, then the script's
own directory.

`-Clean`

Run cleanup before the normal workflow.

`-Build`

Build required images before the normal workflow.

`-Down`

Remove the selected runtime container(s). If used without `-Up`, the script
exits after removal.

`-Up`

Explicitly run the normal configure/start/register/test workflow. This is the
default when `-Down` is not supplied. Use `-Down -Up` to recreate from scratch.

### Node-only parameter

`-Target`

Selects which node service to handle:

```powershell
-Target node
-Target node2
-Target both
```

## GUI workflow

Build the GUI image when needed:

```powershell
docker compose build gui
```

Start the GUI:

```powershell
docker compose up -d gui
```

Start the second GUI:

```powershell
docker compose up -d gui2
```

Check status and logs:

```powershell
docker compose ps gui gui2
docker compose logs --no-color --tail 120 gui
docker compose logs --no-color --tail 120 gui2
```

Open from Windows:

```text
https://127.0.0.1:8443
http://127.0.0.1:8484
https://127.0.0.1:8444
http://127.0.0.1:8485
```

`gui2` reuses the same `fedbiomed/vpn-gui` image as `gui`, so building `gui`
is enough.

## Manual endpoint edits

If not using the PowerShell scripts, update these files consistently.

For the vpnserver main config:

```text
envs/vpn/docker/vpnserver/run_mounts/config/config.env
```

Set:

```bash
export VPN_SERVER_PUBLIC_ADDR=<your-host-ip>
export VPN_SERVER_PORT=51820
```

For the generated server-side node peer config:

```text
envs/vpn/docker/vpnserver/run_mounts/config/config_peers/node/<peer-id>/config.env
```

Set:

```bash
export VPN_SERVER_ENDPOINT=<your-host-ip>:51820
```

For the node-side config:

```text
envs/vpn/docker/node/run_mounts/config/config.env
```

Set:

```bash
export VPN_SERVER_ENDPOINT=<your-host-ip>:51820
```

Then recreate the node so it reloads `config.env` instead of stale
`wg0.conf`, and register the live public key on vpnserver:

```powershell
docker compose rm -sf node
Remove-Item -Recurse -Force ".\node\run_mounts\config\wireguard" -ErrorAction SilentlyContinue
docker compose up -d node
Start-Sleep -Seconds 5

$pubkey = docker compose exec -T node wg show wg0 public-key
$pubkey = $pubkey.Trim()

docker compose exec -T vpnserver python ./vpn/bin/configure_peer.py remove node <peer-id>
docker compose exec -T vpnserver python ./vpn/bin/configure_peer.py add node <peer-id> $pubkey
```

Check that vpnserver registered the same VPN IP that the node uses:

```powershell
docker compose exec -T node ip -4 addr show wg0
docker compose exec -T node wg show wg0 public-key
docker compose exec -T vpnserver wg show wg0
```

## Useful manual checks

```powershell
docker compose ps
docker compose logs --no-color --tail 120 vpnserver
docker compose logs --no-color --tail 120 researcher
docker compose logs --no-color --tail 120 node
docker compose logs --no-color --tail 120 node2
```

```powershell
docker compose exec -T researcher wg show wg0
docker compose exec -T node wg show wg0
docker compose exec -T node2 wg show wg0
docker compose exec -T vpnserver wg show wg0
```
