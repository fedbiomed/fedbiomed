param(
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

$ResearcherContainer = "fedbiomed-vpn-researcher-$InstanceId"

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

function Ensure-Dockerignore {
    $dockerignore = Join-Path $RepoRoot ".dockerignore"
    $requiredLines = @(
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        "htmlcov",
        "build",
        "dist",
        "*.egg-info",
        "fedbiomed_gui/ui/node_modules",
        "**/node_modules",
        "envs/vpn/docker/*/run_mounts",
        "envs/vpn/docker/*/run_mounts/**"
    )

    $existing = ""
    if (Test-Path $dockerignore) {
        $existing = [IO.File]::ReadAllText($dockerignore)
    }

    $lines = New-Object System.Collections.Generic.List[string]
    if ($existing.Length -gt 0) {
        foreach ($line in (($existing -replace "`r`n", "`n" -replace "`r", "`n") -split "`n")) {
            if ($line.Length -gt 0) {
                $lines.Add($line)
            }
        }
    }

    foreach ($line in $requiredLines) {
        if (-not $lines.Contains($line)) {
            $lines.Add($line)
        }
    }

    Write-LinuxTextNoBom -Path $dockerignore -Text (($lines -join "`n") + "`n")
}

function Convert-VpnScriptsToLinuxLineEndings {
    $files = Get-ChildItem (Join-Path $RepoRoot "envs") -Recurse -File -Include "*.bash", "*.sh"
    foreach ($file in $files) {
        Convert-FileToLinuxNoBom -Path $file.FullName
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

function Get-VpnServerPublicAddrForConfig {
    $addr = Get-VpnServerPublicAddr

    if ($addr -match "^\[.+\]:\d+$") {
        return ($addr -replace "^\[(.+)\]:\d+$", '[$1]')
    }

    if ($addr -match "^[^:]+:\d+$") {
        return ($addr -replace ":\d+$", "")
    }

    if ($addr.Contains(":") -and -not ($addr.StartsWith("[") -and $addr.EndsWith("]"))) {
        return "[$addr]"
    }

    return $addr
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

function Down-Researcher {
    Write-Host "== removing researcher container =="
    Invoke-Docker compose rm -sf researcher
}

function Clean-ImageEquivalent {
    Write-Host "== clean image equivalent =="

    Invoke-DockerAllowFailure compose rm -sf vpnserver researcher node node2 gui gui2 base basenode

    foreach ($networkSuffix in @("vpnserver", "researcher", "node", "node2", "gui", "gui2", "misc")) {
        Invoke-DockerAllowFailure network rm "fedbiomed_${networkSuffix}_$InstanceId"
    }

    foreach ($container in @("vpnserver", "researcher", "node", "node2")) {
        Remove-Item -Recurse -Force ".\$container\run_mounts\config\wireguard" -ErrorAction SilentlyContinue
        Remove-Item -Force ".\$container\run_mounts\config\config.env" -ErrorAction SilentlyContinue
    }

    Remove-Item -Recurse -Force ".\vpnserver\run_mounts\config\config_peers" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force ".\vpnserver\run_mounts\config\ip_assign" -ErrorAction SilentlyContinue

    Invoke-DockerAllowFailure image rm -f `
        "fedbiomed/vpn-vpnserver:$VersionTag" `
        "fedbiomed/vpn-researcher:$VersionTag" `
        "fedbiomed/vpn-node:$VersionTag" `
        "fedbiomed/vpn-gui:$VersionTag" `
        "fedbiomed/vpn-base:$VersionTag" `
        "fedbiomed/vpn-basenode:$VersionTag"

    Invoke-Docker image prune -f
}

function Build-Equivalent {
    Write-Host "== build vpnserver researcher equivalent =="

    Invoke-Docker compose build base

    Invoke-Docker compose rm -sf vpnserver
    Invoke-Docker compose build vpnserver

    Invoke-Docker compose rm -sf researcher
    Invoke-Docker compose build researcher
}

function Ensure-VpnserverRunning {
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
        Invoke-DockerAllowFailure compose logs --no-color vpnserver
        throw "vpnserver did not stay running"
    }
}

function Configure-ResearcherEquivalent {
    Write-Host "== configure researcher equivalent =="

    $sample = ".\vpnserver\run_mounts\config\config.env.sample"
    $config = ".\vpnserver\run_mounts\config\config.env"

    if (-not (Test-Path $sample)) {
        throw "Missing vpnserver sample config: $sample"
    }

    $text = [IO.File]::ReadAllText((Resolve-Path $sample))
    $text = $text -replace "replace_with.myserver.mydomain", (Get-VpnServerPublicAddrForConfig)
    Write-LinuxTextNoBom -Path $config -Text $text

    Ensure-VpnserverRunning

    Remove-Item -Recurse -Force ".\vpnserver\run_mounts\config\config_peers\researcher\researcher1" -ErrorAction SilentlyContinue

    Invoke-Docker compose exec -T vpnserver bash -lc "python ./vpn/bin/configure_peer.py genconf researcher researcher1"

    $generated = ".\vpnserver\run_mounts\config\config_peers\researcher\researcher1\config.env"
    if (-not (Test-Path $generated)) {
        Invoke-DockerAllowFailure compose logs --no-color vpnserver
        throw "vpnserver did not generate researcher config.env"
    }

    Remove-Item -Recurse -Force ".\researcher\run_mounts\config\wireguard" -ErrorAction SilentlyContinue
    Copy-Item $generated ".\researcher\run_mounts\config\config.env" -Force

    $researcherConfig = ".\researcher\run_mounts\config\config.env"
    $researcherConfigText = [IO.File]::ReadAllText((Resolve-Path $researcherConfig))
    $researcherConfigText = $researcherConfigText -replace "^export VPN_SERVER_ENDPOINT=.*", "export VPN_SERVER_ENDPOINT=$(Get-VpnServerEndpoint)"
    Write-LinuxTextNoBom -Path $researcherConfig -Text $researcherConfigText

    Convert-FileToLinuxNoBom -Path $generated
}

function Start-ResearcherEquivalent {
    Write-Host "== start researcher equivalent =="

    Ensure-VpnserverRunning

    Invoke-Docker compose rm -sf researcher
    Remove-Item -Recurse -Force ".\researcher\run_mounts\config\wireguard" -ErrorAction SilentlyContinue

    Invoke-Docker compose create researcher

    Invoke-Docker start $ResearcherContainer
    Start-Sleep -Seconds 5

    $researcherState = docker inspect -f "{{.State.Status}}" $ResearcherContainer 2>$null
    if ($researcherState -ne "running") {
        Invoke-DockerAllowFailure compose logs --no-color --tail 120 researcher
        throw "researcher did not stay running"
    }

    $pubkey = (& docker compose exec -T researcher wg show wg0 public-key).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pubkey)) {
        throw "could not read researcher WireGuard public key"
    }

    Invoke-Docker compose exec -T vpnserver python ./vpn/bin/configure_peer.py remove researcher researcher1
    Invoke-Docker compose exec -T vpnserver python ./vpn/bin/configure_peer.py add researcher researcher1 $pubkey

    Start-Sleep -Seconds 2
}

function Status-Equivalent {
    Write-Host "== status vpnserver researcher equivalent =="

    Invoke-Docker compose ps vpnserver researcher

    Write-Host "== ping VPN server over wg0 =="
    Invoke-DockerAllowFailure compose exec -T researcher ping -n -c 3 -W 1 10.220.0.1

    Write-Host "== researcher wg0 =="
    Invoke-Docker compose exec -T researcher wg show wg0

    Write-Host "== vpnserver wg0 =="
    Invoke-Docker compose exec -T vpnserver wg show wg0
}

Push-Location $ComposeDir
try {
    Write-Host "Repository root: $RepoRoot"
    Write-Host "Compose dir:     $ComposeDir"
    Write-Host "VPN endpoint:    $(Get-VpnServerEndpoint)"

    Ensure-Dockerignore
    Convert-VpnScriptsToLinuxLineEndings

    if ($Down) {
        Down-Researcher
        if (-not $Up) {
            return
        }
    }

    if ($Clean) {
        Clean-ImageEquivalent
    }

    if ($Build) {
        Build-Equivalent
    }

    Configure-ResearcherEquivalent
    Start-ResearcherEquivalent
    Status-Equivalent
}
finally {
    Pop-Location
}
