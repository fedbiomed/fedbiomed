#!/bin/bash

# Fed-BioMed - researcher container launch script
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

trap finish TERM INT QUIT

## TODO : make more general by including in the VPN configuration file and in user environment
export RESEARCHER_SERVER_HOST=10.222.0.2
export RESEARCHER_SERVER_PORT=50051
export SECAGG_INSECURE_VALIDATION=False
su -c "export FBM_RESEARCHER_COMPONENT_ROOT=/fbm-researcher ; \
	conda activate fedbiomed-researcher ; \
    fedbiomed configuration create --root /fbm-researcher --component researcher --exist-ok; cd notebooks ; \
    jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' " $CONTAINER_USER &

# proxy port for TensorBoard
# enables launching TB without `--host` option (thus listening only on `localhost`)
# + `watch` for easy respawn in case of failure
    while true ; do \
        socat TCP-LISTEN:6007,fork,reuseaddr,su=$CONTAINER_USER TCP4:127.0.0.1:6006 ; \
        sleep 1 ; \
    done &

sleep infinity &

wait $!
