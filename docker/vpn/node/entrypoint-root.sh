#!/bin/bash

source /entrypoint-functions.sh

[ -f /config/config.env ] && source /config/config.env

# do we prefer to use wireguard kernel vs userspace version
[ -z "$USE_WG_KERNEL_MOD" ] && USE_WG_KERNEL_MOD=false
echo "info: set USE_WG_KERNEL_MOD='${USE_WG_KERNEL_MOD}'"

# which version of wireguard is running
RUNNING_KERNELWG=false
RUNNING_BORINGTUN=false

# Check which VPN variables are present
echo 'INFO: Checking wireguard required variables -----------------------------------------------------------'
VPN_VARS_MISSING=false
for var in VPN_IP VPN_SUBNET_PREFIX VPN_PRIVATE_KEY VPN_SERVER_ENDPOINT VPN_SERVER_ALLOWED_IPS VPN_SERVER_PUBLIC_KEY VPN_SERVER_PSK
do
    value=$(eval "echo \$$var")
    if [ -z "$value" ]
    then
        VPN_VARS_MISSING=true
        echo "WARNING: VPN variable $var is not set"
    else
        [[ "$var" = 'VPN_SERVER_PSK' || "$var" = 'VPN_PRIVATE_KEY' ]] && value='_HIDDEN_VALUE_'
        echo "info: set $var='$value'"
    fi
done

if [ "$VPN_VARS_MISSING" = "true" ]
then
    echo ""
    echo "WARNING: One or more VPN configuration variables are missing."
    echo "         WireGuard will NOT be started."
    echo ""
    echo "         To enable VPN, restart the container with all of the following"
    echo "         environment variables set:"
    echo ""
    echo "           VPN_IP                 - IP address for this node on the VPN (e.g. 10.8.0.2)"
    echo "           VPN_SUBNET_PREFIX      - Subnet prefix length (e.g. 24)"
    echo "           VPN_PRIVATE_KEY        - WireGuard private key for this node (from configure_peer.py genconf)"
    echo "           VPN_SERVER_ENDPOINT    - VPN server address and port (e.g. vpn.example.com:51820)"
    echo "           VPN_SERVER_ALLOWED_IPS - Allowed IPs routed through the tunnel (e.g. 10.8.0.0/24)"
    echo "           VPN_SERVER_PUBLIC_KEY  - WireGuard public key of the VPN server (44-char base64)"
    echo "           VPN_SERVER_PSK         - Pre-shared key (44-char base64)"
    echo ""
    echo "         These values are produced by running on the VPN server:"
    echo "           configure_peer.py genconf node <node-id>"
    echo ""
    echo "         Example docker run flags:"
    echo "           -e VPN_IP=10.8.0.2"
    echo "           -e VPN_SUBNET_PREFIX=24"
    echo "           -e VPN_PRIVATE_KEY=<44-char-base64>"
    echo "           -e VPN_SERVER_ENDPOINT=vpn.example.com:51820"
    echo "           -e VPN_SERVER_ALLOWED_IPS=10.8.0.0/24"
    echo "           -e VPN_SERVER_PUBLIC_KEY=<44-char-base64>"
    echo "           -e VPN_SERVER_PSK=<44-char-base64>"
    echo ""
else
    echo 'INFO: All VPN variables present - starting wireguard ---------------------------------------------------'
    start_wireguard

    echo 'INFO: Configuring wireguard in the container -----------------------------------------------------------'
    configure_wireguard
fi