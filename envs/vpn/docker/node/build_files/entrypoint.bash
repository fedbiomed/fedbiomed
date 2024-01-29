#!/bin/bash

# Fed-BioMed - node container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint

check_vpn_environ
init_misc_environ
change_path_owner "/fedbiomed" "/home/$CONTAINER_BUILD_USER"
start_wireguard
configure_wireguard

COMMON_DIR="/fedbiomed/envs/common/"
if [ -z "$(ls -1 $COMMON_DIR)" ]; then
  # TODO: test with privilege drop
  # $SETUSER rsync -auxt "/fedbiomed/envs/common_reference/" "$COMMON_DIR"
  rsync -auxt "/fedbiomed/envs/common_reference/" "$COMMON_DIR"
fi


trap finish TERM INT QUIT

# Could not launch node at this step with MQTT communications because VPN is not yet fully established 
# TODO: refactor to launch gRPC communications node here ?

sleep infinity &

wait $!
