#!/bin/bash

# Fed-BioMed - mosquitto container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point
# - drops privileges to mosquitto:mosquitto at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint

check_vpn_environ
init_misc_environ
start_wireguard
configure_wireguard

trap finish TERM INT QUIT

#sleep infinity &
# 
# mosquitto drops privilege to mosquitto:mosquitto by default
# as the account exists in the eclipse-mosquitto image :
# no need to "su -c" (wont work anyway, this account prevents logging)
/usr/sbin/mosquitto -c /mosquitto-no-auth.conf &
wait $!
