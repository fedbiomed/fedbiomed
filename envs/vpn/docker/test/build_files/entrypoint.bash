#!/bin/bash

# Fed-BioMed - test container launch script
# - launched as root to handle VPN
# - may drop privileges to CONTAINER_USER at some point

set -x 

# read functions
source /entrypoint_functions.bash

# read config.env
source ~/bashrc_entrypoint

check_vpn_environ
init_misc_environ
start_wireguard
configure_wireguard

trap finish TERM INT QUIT

# ## TODO : make more general by including in the VPN configuration file and in user environment
# export MQTT_BROKER=10.220.0.2
# export MQTT_BROKER_PORT=1883
# export UPLOADS_URL="http://10.220.0.3:8000/upload/"
# export PYTHONPATH=/fedbiomed
# su -c "export PATH=${PATH} ; eval $(conda shell.bash hook) ; \
#     conda activate fedbiomed-researcher ; cd notebooks ; \
#     jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' " $CONTAINER_USER &
# 
# # proxy port for TensorBoard
# # enables launching TB without `--host` option (thus listening only on `localhost`)
# # + `watch` for easy respawn in case of failure
#     while true ; do \
#         socat TCP-LISTEN:6007,fork,reuseaddr,su=$CONTAINER_USER TCP4:127.0.0.1:6006 ; \
#         sleep 1 ; \
#     done &

sleep infinity &

wait $!
