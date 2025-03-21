#!/bin/bash

# Fed-BioMed - VPN server container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint

# content of VPN environ is not the same for VPN server
check_vpn_environ 'VPN_IP VPN_SUBNET_PREFIX VPN_MANAGEMENT_IP_ASSIGN VPN_NODE_IP_ASSIGN VPN_RESEARCHER_IP_ASSIGN VPN_SERVER_PUBLIC_ADDR VPN_SERVER_PORT'

init_misc_environ
start_wireguard
configure_wireguard
change_path_owner "" "/home/$CONTAINER_BUILD_USER"

trap finish TERM INT QUIT


# VPN server additional directories
for dir in ip_assign config_peers
do
    $SETUSER mkdir -p $CONFIG_DIR/$dir
    if [ "$?" -ne 0 ]
    then
        echo "CRITICAL: could not create vpn server config directory ${CONFIG_DIR}/$dir"
        exit 1
    fi
done

# VPN server setup
wg set wg0 listen-port $VPN_SERVER_PORT
if [ "$?" -ne 0 ]
then 
    echo "CRITICAL: could not set wireguard server listening port"
    exit 1
fi

echo "VPN server container is ready"
sleep infinity &
wait $!
