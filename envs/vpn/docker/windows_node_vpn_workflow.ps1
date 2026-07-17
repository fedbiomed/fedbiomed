param(
    [ValidateSet("node", "node2", "both")]
    [string]$Target = "node",
    [string]$VpnServerPublicAddr = "",
    [string]$VersionTag = "6.3.3",
    [string]$InstanceId = "default",
    [string]$ContainerUid = "1000",
    [string]$ContainerGid = "1000",
    [string]$ContainerUser = $env:USERNAME,
    [string]$ContainerGroup = $env:USERNAME,
    [string]$ComposeDir = $env:COMPOSE_DIR,
    [switch]$Clean,
    [switch]$Build,
    [switch]$Down,
    [switch]$Up
)

$ErrorActionPreference = "Stop"

$ComposeDir = if ([string]::IsNullOrWhiteSpace($ComposeDir)) { $PSScriptRoot } else { (Resolve-Path $ComposeDir).Path }
$RepoRoot = (Resolve-Path (Join-Path $ComposeDir "..\..\..")).Path

$env:COMPOSE_DIR = $ComposeDir
$env:FBM_CONTAINER_VERSION_TAG = $VersionTag
$env:FBM_CONTAINER_INSTANCE_ID = $InstanceId
$env:CONTAINER_UID = $ContainerUid
$env:CONTAINER_GID = $ContainerGid
$env:CONTAINER_USER = $ContainerUser
$env:CONTAINER_GROUP = $ContainerGroup

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Invoke-DockerAllowFailure {
    & docker @args
}

function Write-LinuxTextNoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $normalized = $Text.TrimStart([char]0xFEFF)
    $normalized = $normalized -replace "`r`n", "`n" -replace "`r", "`n"
    [IO.File]::WriteAllText($Path, $normalized, [Text.UTF8Encoding]::new($false))
}

function Convert-FileToLinuxNoBom {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-Path $Path) {
        Write-LinuxTextNoBom -Path $Path -Text ([IO.File]::ReadAllText($Path))
    }
}

function Get-DefaultHostIPv4 {
    $routes = Get-NetRoute -AddressFamily IPv4 -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue |
        Sort-Object RouteMetric, InterfaceMetric

    foreach ($route in $routes) {
        $ip = Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $route.InterfaceIndex -ErrorAction SilentlyContinue |
            Where-Object {
                $_.IPAddress -notlike "127.*" -and
                $_.IPAddress -notlike "169.254.*" -and
                $_.PrefixOrigin -ne "WellKnown"
            } |
            Select-Object -First 1 -ExpandProperty IPAddress

        if (-not [string]::IsNullOrWhiteSpace($ip)) {
            return $ip
        }
    }

    $fallback = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Sort-Object @{ Expression = { $_.InterfaceAlias -like "vEthernet*" } }, InterfaceMetric |
        Select-Object -First 1 -ExpandProperty IPAddress

    if ([string]::IsNullOrWhiteSpace($fallback)) {
        throw "Could not auto-detect a host IPv4 address. Pass -VpnServerPublicAddr explicitly."
    }

    return $fallback
}

function Get-VpnServerPublicAddr {
    if (-not [string]::IsNullOrWhiteSpace($VpnServerPublicAddr)) {
        return $VpnServerPublicAddr.Trim()
    }

    return Get-DefaultHostIPv4
}

function Get-VpnServerEndpoint {
    $addr = Get-VpnServerPublicAddr

    if ($addr -match "^\[.+\]:\d+$" -or $addr -match "^[^:]+:\d+$") {
        return $addr
    }

    if ($addr.Contains(":") -and -not ($addr.StartsWith("[") -and $addr.EndsWith("]"))) {
        $addr = "[$addr]"
    }

    return "${addr}:51820"
}

function Get-SelectedServices {
    switch ($Target) {
        "node" { return @("node") }
        "node2" { return @("node2") }
        "both" { return @("node", "node2") }
    }
}

function Get-PeerTag {
    param([Parameter(Mandatory = $true)][string]$Service)

    if ($Service -eq "node2") {
        return "NODE2TAG"
    }

    return "NODETAG"
}

function Down-Nodes {
    Write-Host "== removing selected node container(s) =="
    foreach ($service in Get-SelectedServices) {
        Invoke-Docker compose rm -sf $service
    }
}

function Clean-Nodes {
    Write-Host "== cleaning selected node config(s) =="
    Down-Nodes

    foreach ($service in Get-SelectedServices) {
        $peerTag = Get-PeerTag -Service $service
        Remove-Item -Recurse -Force ".\$service\run_mounts\config\wireguard" -ErrorAction SilentlyContinue
        Remove-Item -Force ".\$service\run_mounts\config\config.env" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force ".\vpnserver\run_mounts\config\config_peers\node\$peerTag" -ErrorAction SilentlyContinue
    }
}

function Build-NodeImage {
    Write-Host "== building shared node image =="
    Invoke-Docker compose build basenode
    Invoke-Docker compose build node
}

function Start-Vpnserver {
    Write-Host "== ensuring vpnserver is running =="

    $existingState = docker inspect -f "{{.State.Status}}" "fedbiomed-vpn-vpnserver-$InstanceId" 2>$null
    if ($existingState -eq "running") {
        Write-Host "vpnserver is already running"
        return
    }

    Invoke-Docker compose up -d vpnserver
    Start-Sleep -Seconds 3

    $vpnserverState = docker inspect -f "{{.State.Status}}" "fedbiomed-vpn-vpnserver-$InstanceId" 2>$null
    if ($vpnserverState -ne "running") {
        Invoke-DockerAllowFailure compose logs --no-color --tail 120 vpnserver
        throw "vpnserver did not stay running"
    }
}

function Setup-Node {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Service,
        [Parameter(Mandatory = $true)]
        [string]$PeerTag
    )

    $containerName = "fedbiomed-vpn-$Service-$InstanceId"
    $endpoint = Get-VpnServerEndpoint
    $configDir = Join-Path $ComposeDir "$Service\run_mounts\config"
    $targetConfig = Join-Path $configDir "config.env"
    $generatedConfig = Join-Path $ComposeDir "vpnserver\run_mounts\config\config_peers\node\$PeerTag\config.env"
    $wireguardDir = Join-Path $configDir "wireguard"
    $tmpDir = Join-Path $ComposeDir "tmp"
    $publicKeyFile = Join-Path $tmpDir "publickey-${Service}side"

    Write-Host "== configuring $Service as peer $PeerTag =="

    New-Item -ItemType Directory -Force $configDir | Out-Null
    New-Item -ItemType Directory -Force $tmpDir | Out-Null

    Remove-Item -Recurse -Force (Split-Path $generatedConfig -Parent) -ErrorAction SilentlyContinue
    Invoke-Docker compose exec -T vpnserver bash -lc "python ./vpn/bin/configure_peer.py genconf node $PeerTag"

    if (-not (Test-Path $generatedConfig)) {
        Invoke-DockerAllowFailure compose logs --no-color --tail 120 vpnserver
        throw "vpnserver did not generate $generatedConfig"
    }

    Copy-Item $generatedConfig $targetConfig -Force

    $configText = [IO.File]::ReadAllText($targetConfig)
    $configText = $configText -replace "^export VPN_SERVER_ENDPOINT=.*", "export VPN_SERVER_ENDPOINT=$endpoint"
    Write-LinuxTextNoBom -Path $targetConfig -Text $configText

    $generatedConfigText = [IO.File]::ReadAllText($generatedConfig)
    $generatedConfigText = $generatedConfigText -replace "^export VPN_SERVER_ENDPOINT=.*", "export VPN_SERVER_ENDPOINT=$endpoint"
    Write-LinuxTextNoBom -Path $generatedConfig -Text $generatedConfigText

    Invoke-Docker compose rm -sf $Service
    Remove-Item -Recurse -Force $wireguardDir -ErrorAction SilentlyContinue

    Invoke-Docker compose create $Service

    Invoke-Docker start $containerName
    Start-Sleep -Seconds 5

    $nodeState = docker inspect -f "{{.State.Status}}" $containerName 2>$null
    if ($nodeState -ne "running") {
        Invoke-DockerAllowFailure compose logs --no-color --tail 120 $Service
        throw "$Service did not stay running"
    }

    $pubkey = (& docker compose exec -T $Service wg show wg0 public-key).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pubkey)) {
        throw "could not read $Service WireGuard public key"
    }

    [IO.File]::WriteAllText($publicKeyFile, $pubkey, [Text.UTF8Encoding]::new($false))

    Invoke-Docker compose exec -T vpnserver python ./vpn/bin/configure_peer.py remove node $PeerTag
    Invoke-Docker compose exec -T vpnserver python ./vpn/bin/configure_peer.py add node $PeerTag $pubkey

    Write-Host "== testing $Service VPN =="
    Invoke-DockerAllowFailure compose exec -T $Service ping -n -c 3 -W 1 10.220.0.1

    Write-Host "== $Service wg0 =="
    Invoke-Docker compose exec -T $Service wg show wg0

    Write-Host "== vpnserver wg0 =="
    Invoke-Docker compose exec -T vpnserver wg show wg0
}

Push-Location $ComposeDir
try {
    Write-Host "Repository root: $RepoRoot"
    Write-Host "Compose dir:     $ComposeDir"
    Write-Host "VPN endpoint:    $(Get-VpnServerEndpoint)"

    if ($Down) {
        Down-Nodes
        if (-not $Up) {
            return
        }
    }

    if ($Clean) {
        Clean-Nodes
    }

    if ($Build) {
        Build-NodeImage
    }

    Start-Vpnserver

    switch ($Target) {
        "node" {
            Setup-Node -Service "node" -PeerTag "NODETAG"
        }
        "node2" {
            Setup-Node -Service "node2" -PeerTag "NODE2TAG"
        }
        "both" {
            Setup-Node -Service "node" -PeerTag "NODETAG"
            Setup-Node -Service "node2" -PeerTag "NODE2TAG"
        }
    }
}
finally {
    Pop-Location
}
